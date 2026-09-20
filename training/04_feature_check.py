"""Which numbers make the models work? A quick diagnosis.

  python training/04_feature_check.py

It trains each model three times: on all 12 numbers, on the 5 numbers that compare
the resume with the job ("match"), and on the 7 numbers that are only lengths and
counts ("other"). If "other" does nearly as well as "match", the models are learning
shortcuts in the dataset instead of how well a resume fits a job."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from features import FEATURE_NAMES, MATCH_FEATURES, OTHER_FEATURES   # noqa: E402

train = pd.read_csv("data/features_train.csv").dropna()
test = pd.read_csv("data/features_test.csv").dropna()
y_train, y_test = train["label"].astype(int), test["label"].astype(int)

feature_sets = {
    "all (12 numbers)": FEATURE_NAMES,
    "match only (5)": MATCH_FEATURES,
    "other only (7)": OTHER_FEATURES,
}


def make_models():
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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for set_name, columns in feature_sets.items():
    for model_name, model in make_models().items():
        print(f"  {set_name} / {model_name} ...")
        cv_f1 = cross_val_score(model, train[columns], y_train, cv=cv, scoring="f1_macro").mean()
        if model_name == "XGBoost":        # XGBoost has no class_weight option
            from sklearn.utils.class_weight import compute_sample_weight
            model.fit(train[columns], y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(train[columns], y_train)
        proba = model.predict_proba(test[columns])
        predicted = proba.argmax(axis=1)
        rows.append({
            "features": set_name,
            "model": model_name,
            "cv_f1_macro": cv_f1,
            "test_accuracy": accuracy_score(y_test, predicted),
            "test_f1_macro": f1_score(y_test, predicted, average="macro"),
            "test_spearman": spearmanr(proba @ np.array([0, 50, 100]), y_test)[0],
        })

table = pd.DataFrame(rows)
print("\n=== RESULTS ===")
print(table.round(3).to_string(index=False))
Path("reports").mkdir(exist_ok=True)
table.to_csv("reports/feature_check.csv", index=False)

best = table.groupby("features")["cv_f1_macro"].max()
print("\nBest cross-validation F1 per feature set:")
print(best.round(3).to_string())
gap = best["other only (7)"] - best["match only (5)"]
print()
if gap > 0.02:
    print("The length/count numbers beat the resume-vs-job numbers. The dataset labels seem to")
    print("depend on things like text length, which is not what your website should reward.")
elif best["other only (7)"] > 0.9 * best["match only (5)"]:
    print("The length/count numbers alone are almost as good as the match numbers.")
    print("Part of the score comes from shortcuts, not from real resume-job matching.")
else:
    print("The match numbers clearly beat the length/count numbers. Good sign.")
