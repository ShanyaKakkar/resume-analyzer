"""Do the numbers work once we compare resumes for the SAME job description?

  python training/06_within_job_check.py

Every job description in the dataset is paired with many resumes, and some jobs make
all resumes look similar while others make all of them look different. That job-specific
shift can hide the real signal. This script measures how much signal is left when the
shift is removed. It needs no language model, so it runs in about a minute."""
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from features import FEATURE_NAMES, MATCH_FEATURES   # noqa: E402


warnings.filterwarnings("ignore")     # constant numbers give harmless warnings


def text_ids(series):
    return series.astype(str).map(lambda t: hashlib.md5(t.strip().encode("utf-8")).hexdigest())


def load(split):
    raw = pd.read_csv(f"data/raw/fit_{split}.csv")
    feat = pd.read_csv(f"data/features_{split}.csv")
    if len(raw) != len(feat):
        sys.exit("The feature files must come from the FULL dataset (no --limit).")
    return feat, text_ids(raw["job_description_text"]).values


train, g_train = load("train")
test, g_test = load("test")
y_train, y_test = train["label"].astype(int), test["label"].astype(int)

print("=== 1. Does each number predict the label? ===")
print("overall = all rows together | within a job = compare resumes of the same job only\n")
rows = []
for name in FEATURE_NAMES:
    overall = spearmanr(train[name], y_train)[0]
    per_job = []
    for job in np.unique(g_train):
        mask = g_train == job
        if mask.sum() >= 10 and y_train[mask].nunique() > 1 and train.loc[mask, name].nunique() > 1:
            per_job.append(spearmanr(train.loc[mask, name], y_train[mask])[0])
    rows.append({"number": name, "overall": overall,
                 "within_a_job": np.mean(per_job) if per_job else np.nan,
                 "jobs_where_positive": np.mean(np.array(per_job) > 0) if per_job else np.nan})
print(pd.DataFrame(rows).round(3).to_string(index=False))


def centered(frame, groups):
    """Subtract each job's average, so only 'better or worse than usual for this job' is left."""
    out = frame[MATCH_FEATURES].copy()
    return out - out.groupby(groups).transform("mean")


def evaluate(X_train, X_test):
    scores = {}
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=40, random_state=42,
            n_jobs=-1, class_weight="balanced_subsample"),
    }
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=GroupKFold(5), groups=g_train,
                             scoring="f1_macro").mean()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        scores[name] = (cv, f1_score(y_test, predicted, average="macro"),
                        accuracy_score(y_test, predicted))
    return scores


print("\n=== 2. What if each job's own shift were removed? (an upper bound) ===")
print("'centered' uses the other resumes of the same job, which a live website will NOT have.")
print("It shows the best we could hope for from a background-comparison approach.\n")
plain = evaluate(train[MATCH_FEATURES], test[MATCH_FEATURES])
shifted = evaluate(centered(train, g_train), centered(test, g_test))
table = []
for label, scores in (("as they are", plain), ("centered per job", shifted)):
    for model, (cv, f1, acc) in scores.items():
        table.append({"match numbers": label, "model": model, "grouped_cv_f1": cv,
                      "test_f1": f1, "test_accuracy": acc})
result = pd.DataFrame(table)
print(result.round(3).to_string(index=False))
Path("reports").mkdir(exist_ok=True)
result.to_csv("reports/within_job_check.csv", index=False)

gain = result[result["match numbers"] == "centered per job"]["test_f1"].max() - \
       result[result["match numbers"] == "as they are"]["test_f1"].max()
print()
if gain > 0.05:
    print(f"Removing each job's own shift raises the test F1 by {gain:.2f}. Worth building:")
    print("compare each new resume with a fixed set of background resumes, and score it")
    print("relative to them.")
else:
    print(f"Removing each job's own shift raises the test F1 by only {gain:.2f}. The numbers")
    print("themselves carry little signal in this dataset, so normalising would not help much.")
