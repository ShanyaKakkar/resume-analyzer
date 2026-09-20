"""Does the same job description (or resume) repeat across many rows?

  python training/05_group_check.py

If it does, ordinary cross-validation cheats: the same job description sits in both the
learning part and the checking part, so a model can memorise it. This script shows the
facts and compares ordinary cross-validation with a fair version (grouped by job
description) and with the untouched test rows."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from features import FEATURE_NAMES, MATCH_FEATURES, OTHER_FEATURES   # noqa: E402


def text_ids(series):
    return series.astype(str).map(lambda t: hashlib.md5(t.strip().encode("utf-8")).hexdigest())


raw_train = pd.read_csv("data/raw/fit_train.csv")
raw_test = pd.read_csv("data/raw/fit_test.csv")
feat_train = pd.read_csv("data/features_train.csv")
feat_test = pd.read_csv("data/features_test.csv")
if len(raw_train) != len(feat_train) or len(raw_test) != len(feat_test):
    sys.exit("The feature files must come from the FULL dataset (no --limit), so that rows line up.")

jd_train, jd_test = text_ids(raw_train["job_description_text"]), text_ids(raw_test["job_description_text"])
cv_train, cv_test = text_ids(raw_train["resume_text"]), text_ids(raw_test["resume_text"])

print("=== THE FACTS ===")
for name, jd, res in (("train", jd_train, cv_train), ("test", jd_test, cv_test)):
    per_jd = jd.value_counts()
    print(f"{name}: {len(jd)} rows | {jd.nunique()} different job descriptions "
          f"(each used {per_jd.mean():.1f} times on average, at most {per_jd.max()}) "
          f"| {res.nunique()} different resumes")
print("job descriptions found in BOTH train and test:", len(set(jd_train) & set(jd_test)))
print("resumes found in BOTH train and test:", len(set(cv_train) & set(cv_test)))
y_train, y_test = feat_train["label"].astype(int), feat_test["label"].astype(int)
print("train labels:", y_train.value_counts().sort_index().to_dict(),
      "| test labels:", y_test.value_counts().sort_index().to_dict(), "(0 No, 1 Potential, 2 Good)")
majority = y_train.value_counts().idxmax()
proportions = y_train.value_counts(normalize=True).sort_index().values
print(f"always guessing the most common label gets {(y_test == majority).mean():.3f} accuracy on test")
print(f"a model that only knows the label proportions has log-loss "
      f"{log_loss(y_test, np.tile(proportions, (len(y_test), 1)), labels=[0, 1, 2]):.3f} "
      f"(a useful model must beat this)")

feature_sets = {"all (12)": FEATURE_NAMES, "match only (5)": MATCH_FEATURES, "other only (7)": OTHER_FEATURES}
weights = compute_sample_weight("balanced", y_train)


def models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42,
            n_jobs=-1, class_weight="balanced_subsample"),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
    }


random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grouped_cv = GroupKFold(n_splits=5)
rows = []
print("\nTraining (a few minutes) ...")
for set_name, columns in feature_sets.items():
    for model_name, model in models().items():
        extra = {"sample_weight": weights} if model_name == "XGBoost" else {}
        ordinary = cross_val_score(model, feat_train[columns], y_train, cv=random_cv,
                                   scoring="f1_macro", params=extra).mean()
        fair = cross_val_score(model, feat_train[columns], y_train, cv=grouped_cv, groups=jd_train,
                               scoring="f1_macro", params=extra).mean()
        model.fit(feat_train[columns], y_train, **extra) if extra else model.fit(feat_train[columns], y_train)
        test_f1 = f1_score(y_test, model.predict(feat_test[columns]), average="macro")
        rows.append({"features": set_name, "model": model_name, "ordinary_cv": ordinary,
                     "grouped_cv": fair, "test": test_f1})
        print(f"  {set_name} / {model_name} done")

table = pd.DataFrame(rows)
print("\n=== MACRO-F1: ordinary CV vs fair (grouped) CV vs test ===")
print(table.round(3).to_string(index=False))
Path("reports").mkdir(exist_ok=True)
table.to_csv("reports/group_check.csv", index=False)

gap = (table["ordinary_cv"] - table["test"]).max()
fair_gap = (table["grouped_cv"] - table["test"]).abs().max()
print()
if gap > 0.08 and fair_gap < gap / 2:
    print("CONFIRMED: ordinary cross-validation was too optimistic (it lets the same job")
    print("descriptions appear on both sides). Grouped cross-validation matches the test rows.")
    print("Use grouped cross-validation to choose your model.")
elif gap > 0.08:
    print("Ordinary cross-validation is too optimistic, and grouped cross-validation still")
    print("differs from the test rows. Send me this table and we will look further.")
else:
    print("Ordinary and grouped cross-validation agree, so repeated rows are not the problem.")
