"""Train a "job family" model: 9 broad families instead of 24 detailed categories.

  python training/11_train_families.py

Many of the 24 categories overlap in real life (Accountant / Finance / Banking, or
Business-Development / Sales / Consultant), and even a person could not tell those resumes
apart. Grouping them into families is a task change, so always report it as one.

Two inputs are compared on exactly the same resumes:
  1. skills only (the Skills section)
  2. the whole resume, without the first 15 words (so a job title at the top cannot
     give the answer away)
Random forest, logistic regression and XGBoost are compared fairly for each."""
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

# Edit this if you disagree with a grouping, then run the script again.
FAMILIES = {
    "Technology & Engineering": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
    "Finance & Accounting": ["FINANCE", "ACCOUNTANT", "BANKING"],
    "Business, Sales & Support": ["BUSINESS-DEVELOPMENT", "SALES", "CONSULTANT", "BPO"],
    "Creative, Media & Communications": ["DESIGNER", "DIGITAL-MEDIA", "ARTS", "APPAREL", "PUBLIC-RELATIONS"],
    "Healthcare & Fitness": ["HEALTHCARE", "FITNESS"],
    "Education": ["TEACHER"],
    "HR & Legal": ["HR", "ADVOCATE"],
    "Hospitality & Aviation": ["CHEF", "AVIATION"],
    "Construction, Automotive & Agriculture": ["CONSTRUCTION", "AUTOMOBILE", "AGRICULTURE"],
}
CATEGORY_TO_FAMILY = {c: f for f, cats in FAMILIES.items() for c in cats}
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
    return " ".join(str(text).split()[n:])


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


df = pd.read_csv(find_csv())[["Resume_str", "Category"]].dropna()
df["Resume_str"] = df["Resume_str"].astype(str)
df = df[df["Resume_str"].str.split().str.len() >= 20].drop_duplicates(subset="Resume_str").reset_index(drop=True)
unknown = sorted(set(df["Category"]) - set(CATEGORY_TO_FAMILY))
if unknown:
    sys.exit(f"These categories are not in FAMILIES: {unknown}. Add them to the dictionary at the top.")
df["family"] = df["Category"].map(CATEGORY_TO_FAMILY)
df["skills"] = df["Resume_str"].map(get_skills_text)
df["n_skills"] = df["skills"].map(lambda s: len(skill_phrases(s)))
df["content"] = df["Resume_str"].map(lambda t: drop_words(t, HEADLINE_WORDS))
usable = df["n_skills"] >= 3
print(f"resumes: {len(df)} | with a readable Skills section: {int(usable.sum())} ({usable.mean():.0%})")

work = df[usable].reset_index(drop=True)
encoder = LabelEncoder()
y = encoder.fit_transform(work["family"])
families = list(encoder.classes_)
print("resumes per family:", ", ".join(f"{f} ({n})" for f, n in work["family"].value_counts().items()))
groups = duplicate_groups(work["Resume_str"])

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(splitter.split(work, y, groups))
y_train, y_test, g_train = y[train_idx], y[test_idx], groups[train_idx]
print(f"train: {len(train_idx)} | test: {len(test_idx)}")
majority = np.bincount(y_test).max() / len(y_test)
weights = compute_sample_weight("balanced", y_train)


def tfidf(max_features):
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8,
                           sublinear_tf=True, max_features=max_features)


def candidates(max_features):
    return {
        "Logistic Regression": (
            make_pipeline(tfidf(max_features), LogisticRegression(max_iter=3000, class_weight="balanced")),
            {"logisticregression__C": [3, 10, 30]}, {}),
        "Random Forest": (
            make_pipeline(tfidf(max_features), RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample")),
            {"randomforestclassifier__max_features": ["sqrt"]}, {}),
        "XGBoost": (
            make_pipeline(tfidf(max_features), XGBClassifier(
                objective="multi:softprob", n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.5, tree_method="hist", n_jobs=-1,
                random_state=42, eval_metric="mlogloss")),
            {"xgbclassifier__max_depth": [4]}, {"xgbclassifier__sample_weight": weights}),
    }


inputs = {
    "skills only": ("skills", 8000),
    "whole resume, first 15 words removed": ("content", 6000),
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
rows, best_models = [], {}
for input_name, (column, max_features) in inputs.items():
    X_train, X_test = work[column].iloc[train_idx], work[column].iloc[test_idx]
    fitted = {}
    for model_name, (pipeline, grid, fit_args) in candidates(max_features).items():
        print(f"\n[{input_name}] tuning {model_name} ...")
        search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=1, refit="f1_macro",
                              scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"})
        search.fit(X_train, y_train, groups=g_train, **fit_args)
        best = search.best_estimator_
        fitted[model_name] = best
        proba = best.predict_proba(X_test)
        predicted = proba.argmax(axis=1)
        top2 = np.mean([label in np.argsort(p)[-2:] for label, p in zip(y_test, proba)])
        i = search.best_index_
        rows.append({
            "input": input_name, "model": model_name,
            "cv_accuracy": search.cv_results_["mean_test_accuracy"][i],
            "cv_f1_macro": search.cv_results_["mean_test_f1_macro"][i],
            "test_accuracy": accuracy_score(y_test, predicted),
            "test_f1_macro": f1_score(y_test, predicted, average="macro"),
            "test_top2_accuracy": top2,
        })
    winner = max((r for r in rows if r["input"] == input_name), key=lambda r: r["cv_f1_macro"])["model"]
    best_models[input_name] = (winner, fitted[winner])

table = pd.DataFrame(rows)
print(f"\n=== RESULTS: {len(families)} job families ===")
print(table.round(3).to_string(index=False))
print(f"\nAlways guessing the biggest family: about {majority:.3f} accuracy.")
table.to_csv(REPORTS_DIR / "family_comparison.csv", index=False)

for input_name, (winner, model) in best_models.items():
    column = inputs[input_name][0]
    predicted = model.predict(work[column].iloc[test_idx])
    print(f"\n[{input_name}] winner: {winner}. F1 per family on the test part:")
    report = classification_report(y_test, predicted, target_names=families, output_dict=True, zero_division=0)
    for family in sorted(families, key=lambda f: report[f]["f1-score"]):
        print(f"  {family}: {report[family]['f1-score']:.2f} ({int(report[family]['support'])} test resumes)")

# ---------- typical skills per family (to explain the answer) ----------
per_resume = work["skills"].map(skill_phrases)
overall = Counter(p for s in per_resume for p in s)
typical = {}
for family in families:
    members = per_resume[work["family"] == family]
    counts = Counter(p for s in members for p in s)
    scored = []
    for phrase, count in counts.items():
        share_in, share_all = count / len(members), overall[phrase] / len(work)
        if count >= 5 and share_in >= 0.03:
            scored.append((share_in / share_all, count, phrase))
    typical[family] = [p for _, _, p in sorted(scored, reverse=True)[:40]]
(MODELS_DIR / "family_typical_skills.json").write_text(json.dumps(typical, indent=1))

# ---------- save both models, trained on all the data they can use ----------
for input_name, (winner, model) in best_models.items():
    column = inputs[input_name][0]
    if column == "content":            # the whole-resume model does not need a Skills section
        data, labels = df, LabelEncoder().fit(work["family"]).transform(df["family"])
    else:
        data, labels = work, y
    final = clone(model)
    args = {"xgbclassifier__sample_weight": compute_sample_weight("balanced", labels)} if winner == "XGBoost" else {}
    final.fit(data[column], labels, **args)
    name = "family_skills_model" if column == "skills" else "family_content_model"
    joblib.dump({"model": final, "classes": families, "name": winner, "families": FAMILIES,
                 "drop_first_words": HEADLINE_WORDS if column == "content" else 0},
                MODELS_DIR / f"{name}.joblib")
    print(f"saved models/{name}.joblib ({winner})")
