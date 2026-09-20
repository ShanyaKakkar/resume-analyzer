"""Train a "best-fit job category" classifier: which of the 24 categories does a resume belong to?

  python training/08_train_category_model.py

Fair-play rules used here (the same lessons as the fit model):
  * duplicate and near-duplicate resumes always stay on the same side of every split
  * models are compared with grouped, stratified cross-validation on the TRAINING part only
  * the untouched TEST part is used once, at the end
  * we also test what happens when the first 15 words (usually the job title at the top
    of the resume) are removed, because a title can give the category away
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--drop-first-words", type=int, default=0,
                    help="remove this many words from the top of every resume before training "
                         "the model that gets saved (0 = keep everything)")
args = parser.parse_args()
HEADLINE_WORDS = 15

MODELS_DIR, REPORTS_DIR = Path("models"), Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def find_csv():
    for path in sorted(Path("data/raw").rglob("*.csv")):
        if path.name.startswith(("fit_", "features_")):
            continue
        try:
            columns = pd.read_csv(path, nrows=1).columns
        except Exception:
            continue
        if "Category" in columns and "Resume_str" in columns:
            return path
    sys.exit("Could not find the Kaggle Resume.csv. Run training/07_check_category_data.py first.")


def drop_words(text, n):
    return " ".join(str(text).split()[n:]) if n else str(text)


def duplicate_groups(texts, threshold=0.9):
    """Give the same group number to resumes that are (nearly) copies of each other."""
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_features=20000).fit_transform(texts)
    similar = linear_kernel(tfidf) > threshold
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(similar, k=1))):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(texts))])


# ---------- load and clean ----------
csv_path = find_csv()
df = pd.read_csv(csv_path)[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str)
before = len(df)
df = df[df["Resume_str"].str.split().str.len() >= 20]
df = df.drop_duplicates(subset="Resume_str").reset_index(drop=True)
print(f"{csv_path}: {before} rows, {len(df)} kept "
      f"(removed {before - len(df)} empty or exactly duplicated resumes)")

encoder = LabelEncoder()
y = encoder.fit_transform(df["Category"])
classes = list(encoder.classes_)
texts = df["Resume_str"]

groups = duplicate_groups(texts)
in_groups = int((pd.Series(groups).map(pd.Series(groups).value_counts()) > 1).sum())
print(f"near-duplicate check: {in_groups} resumes have a near-copy (kept on the same side of every split)")

# ---------- one fixed train / test split (20% test) ----------
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(texts, y, groups))
X_train, X_test = texts.iloc[train_idx], texts.iloc[test_idx]
y_train, y_test, g_train = y[train_idx], y[test_idx], groups[train_idx]
print(f"train: {len(train_idx)} resumes | test: {len(test_idx)} resumes | categories: {len(classes)}")
weights = compute_sample_weight("balanced", y_train)


def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.8,
                           sublinear_tf=True, max_features=6000)


candidates = {
    "Logistic Regression": (
        make_pipeline(tfidf(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        {"logisticregression__C": [3, 10, 30]}, {}),
    "Random Forest": (
        make_pipeline(tfidf(), RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
        {"randomforestclassifier__max_features": ["sqrt"]}, {}),
    "XGBoost": (
        make_pipeline(tfidf(), XGBClassifier(
            objective="multi:softprob", n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=-1,
            random_state=42, eval_metric="mlogloss")),
        {"xgbclassifier__max_depth": [4]}, {"xgbclassifier__sample_weight": weights}),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
rows, fitted = [], {}
for name, (pipeline, grid, fit_args) in candidates.items():
    print(f"\nTuning {name} (this can take several minutes) ...")
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=1, refit="f1_macro",
                          scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"})
    search.fit(X_train, y_train, groups=g_train, **fit_args)
    best = search.best_estimator_
    fitted[name] = best
    predicted = best.predict(X_test)
    i = search.best_index_
    rows.append({
        "model": name,
        "cv_accuracy": search.cv_results_["mean_test_accuracy"][i],
        "cv_f1_macro": search.cv_results_["mean_test_f1_macro"][i],
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_f1_macro": f1_score(y_test, predicted, average="macro"),
        "best_settings": search.best_params_,
    })
    print("  best settings:", search.best_params_)

table = pd.DataFrame(rows)
print("\n=== RESULTS (24 categories) ===")
print(table.drop(columns="best_settings").round(3).to_string(index=False))
print(f"\nFor comparison, always guessing the most common category gets about "
      f"{np.bincount(y_test).max() / len(y_test):.3f} accuracy.")
table.to_csv(REPORTS_DIR / "category_comparison.csv", index=False)

winner = table.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
print(f"\nWinner (best cross-validation F1): {winner}")
best_model = fitted[winner]
predicted = best_model.predict(X_test)

# ---------- what is confused with what ----------
report = classification_report(y_test, predicted, target_names=classes, output_dict=True, zero_division=0)
weakest = sorted(classes, key=lambda c: report[c]["f1-score"])[:5]
print("\nweakest categories (F1 on the test part):")
for c in weakest:
    print(f"  {c}: F1 {report[c]['f1-score']:.2f} ({int(report[c]['support'])} test resumes)")
matrix = confusion_matrix(y_test, predicted, labels=range(len(classes)))
off = matrix.copy()
np.fill_diagonal(off, 0)
print("\nmost common mix-ups (actual -> predicted):")
for flat in np.argsort(off, axis=None)[::-1][:5]:
    a, p = divmod(flat, len(classes))
    if off[a, p]:
        print(f"  {classes[a]} -> {classes[p]}: {off[a, p]} resumes")

# ---------- does the title at the top give the answer away? ----------
without_headline = clone(best_model)
fit_args = {}
if winner == "XGBoost":
    fit_args = {"xgbclassifier__sample_weight": weights}
without_headline.fit(X_train.map(lambda t: drop_words(t, HEADLINE_WORDS)), y_train, **fit_args)
alt = without_headline.predict(X_test.map(lambda t: drop_words(t, HEADLINE_WORDS)))
print(f"\nHEADLINE CHECK for {winner}:")
print(f"  test accuracy with the full resume text:           {accuracy_score(y_test, predicted):.3f}")
print(f"  test accuracy with the first {HEADLINE_WORDS} words removed: {accuracy_score(y_test, alt):.3f}")
print("  Report both. The second number is the safer one for resumes without a title line.")

# ---------- chart ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(normalised, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_title(f"{winner}: share of each actual category predicted as each category")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "category_confusion.png", dpi=150)
    plt.close(fig)
    print("\nChart saved: reports/category_confusion.png")
except ImportError:
    pass

# ---------- save the model (trained on all the data) ----------
final = clone(best_model)
final_texts = texts.map(lambda t: drop_words(t, args.drop_first_words))
final_args = {}
if winner == "XGBoost":
    final_args = {"xgbclassifier__sample_weight": compute_sample_weight("balanced", y)}
final.fit(final_texts, y, **final_args)
joblib.dump({"model": final, "classes": classes, "name": winner,
             "drop_first_words": args.drop_first_words}, MODELS_DIR / "category_model.joblib")
(MODELS_DIR / "category_model_info.json").write_text(json.dumps({
    "winner": winner, "classes": classes, "drop_first_words": args.drop_first_words,
    "results": json.loads(table.to_json(orient="records")),
}, indent=2))
print(f"Saved the winner (trained on all {len(y)} resumes, first {args.drop_first_words} words "
      f"removed) to models/category_model.joblib")
