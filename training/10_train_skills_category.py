"""Train the "skills -> job category" model.

  python training/10_train_skills_category.py

The model sees ONLY the Skills section of each resume: no job title, no summary, no
experience text. It compares the resume's skills with the skills typical for each of the
24 categories. Random forest, logistic regression and XGBoost are compared fairly."""
import json
import sys
from collections import Counter
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from skills_text import get_skills_text, skill_phrases   # noqa: E402

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


def duplicate_groups(texts, threshold=0.9):
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


# ---------- load, find the Skills sections ----------
df = pd.read_csv(find_csv())[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str)
df = df.drop_duplicates(subset="Resume_str").reset_index(drop=True)
df["skills"] = df["Resume_str"].map(get_skills_text)
df["n_skills"] = df["skills"].map(lambda s: len(skill_phrases(s)))
usable = df["n_skills"] >= 3
print(f"resumes: {len(df)} | with a readable Skills section (3 or more skills): "
      f"{int(usable.sum())} ({usable.mean():.0%})")
print("the model is trained and tested on these; resumes without a Skills section are left out")
without = df[~usable]["Category"].value_counts()
print(f"left out: {len(without)} categories affected, most: "
      + ", ".join(f"{c} ({n})" for c, n in without.head(4).items()))
df = df[usable].reset_index(drop=True)

# categories with too few usable resumes cannot be learned or tested fairly
counts = df["Category"].value_counts()
small = counts[counts < 15].index.tolist()
if small:
    print("dropping categories with fewer than 15 usable resumes:", ", ".join(small))
    df = df[~df["Category"].isin(small)].reset_index(drop=True)

encoder = LabelEncoder()
y = encoder.fit_transform(df["Category"])
classes = list(encoder.classes_)
texts = df["skills"]
groups = duplicate_groups(df["Resume_str"])
print(f"training on {len(df)} resumes in {len(classes)} categories; "
      f"average skills per resume: {int(df['n_skills'].mean())}")

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(texts, y, groups))
X_train, X_test = texts.iloc[train_idx], texts.iloc[test_idx]
y_train, y_test, g_train = y[train_idx], y[test_idx], groups[train_idx]
print(f"train: {len(train_idx)} | test: {len(test_idx)}")
weights = compute_sample_weight("balanced", y_train)


def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8,
                           sublinear_tf=True, max_features=8000)


candidates = {
    "Logistic Regression": (
        make_pipeline(tfidf(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        {"logisticregression__C": [3, 10, 30, 100]}, {}),
    "Random Forest": (
        make_pipeline(tfidf(), RandomForestClassifier(
            n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
        {"randomforestclassifier__max_features": ["sqrt"]}, {}),
    "XGBoost": (
        make_pipeline(tfidf(), XGBClassifier(
            objective="multi:softprob", n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=-1,
            random_state=42, eval_metric="mlogloss")),
        {"xgbclassifier__max_depth": [4]}, {"xgbclassifier__sample_weight": weights}),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
rows, fitted = [], {}
for name, (pipeline, grid, fit_args) in candidates.items():
    print(f"\nTuning {name} ...")
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=1, refit="f1_macro",
                          scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"})
    search.fit(X_train, y_train, groups=g_train, **fit_args)
    best = search.best_estimator_
    fitted[name] = best
    proba = best.predict_proba(X_test)
    predicted = proba.argmax(axis=1)
    top3 = np.mean([label in np.argsort(p)[-3:] for label, p in zip(y_test, proba)])
    i = search.best_index_
    rows.append({
        "model": name,
        "cv_accuracy": search.cv_results_["mean_test_accuracy"][i],
        "cv_f1_macro": search.cv_results_["mean_test_f1_macro"][i],
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_f1_macro": f1_score(y_test, predicted, average="macro"),
        "test_top3_accuracy": top3,
        "best_settings": search.best_params_,
    })
    print("  best settings:", search.best_params_)

table = pd.DataFrame(rows)
print(f"\n=== RESULTS: skills only, {len(classes)} categories ===")
print(table.drop(columns="best_settings").round(3).to_string(index=False))
print(f"\nAlways guessing the most common category: about {np.bincount(y_test).max() / len(y_test):.3f} accuracy.")
print("test_top3_accuracy = the right category is among the 3 suggested ones (report it as top-3).")
table.to_csv(REPORTS_DIR / "skills_category_comparison.csv", index=False)

winner = table.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
print(f"\nWinner (best cross-validation F1): {winner}")
predicted = fitted[winner].predict(X_test)
report = classification_report(y_test, predicted, target_names=classes, output_dict=True, zero_division=0)
print("\nweakest categories (F1 on the test part):")
for c in sorted(classes, key=lambda c: report[c]["f1-score"])[:5]:
    print(f"  {c}: F1 {report[c]['f1-score']:.2f} ({int(report[c]['support'])} test resumes)")
matrix = confusion_matrix(y_test, predicted, labels=range(len(classes)))
off = matrix.copy()
np.fill_diagonal(off, 0)
print("\nmost common mix-ups (actual -> predicted):")
for flat in np.argsort(off, axis=None)[::-1][:5]:
    a, p = divmod(flat, len(classes))
    if off[a, p]:
        print(f"  {classes[a]} -> {classes[p]}: {off[a, p]} resumes")

# ---------- typical skills per category (used to explain the answer) ----------
per_resume = df["skills"].map(skill_phrases)
overall = Counter(p for s in per_resume for p in s)
typical = {}
for category in classes:
    members = per_resume[df["Category"] == category]
    counts = Counter(p for s in members for p in s)
    scored = []
    for phrase, count in counts.items():
        share_in, share_all = count / len(members), overall[phrase] / len(df)
        if count >= 5 and share_in >= 0.03:
            scored.append((share_in / share_all, count, phrase))
    typical[category] = [p for _, _, p in sorted(scored, reverse=True)[:40]]

# ---------- save (trained on all usable resumes) ----------
final = clone(fitted[winner])
final_args = {"xgbclassifier__sample_weight": compute_sample_weight("balanced", y)} if winner == "XGBoost" else {}
final.fit(texts, y, **final_args)
joblib.dump({"model": final, "classes": classes, "name": winner},
            MODELS_DIR / "category_skills_model.joblib")
(MODELS_DIR / "category_typical_skills.json").write_text(json.dumps(typical, indent=1))
(MODELS_DIR / "category_skills_model_info.json").write_text(json.dumps({
    "winner": winner, "classes": classes,
    "results": json.loads(table.to_json(orient="records"))}, indent=2))
print(f"\nSaved models/category_skills_model.joblib and models/category_typical_skills.json")
