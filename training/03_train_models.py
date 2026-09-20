"""Train random forest, logistic regression and XGBoost, compare them, keep the best.

  python training/03_train_models.py                  # all 12 numbers
  python training/03_train_models.py --features match # only the resume-vs-job numbers

Fair-play rules used here:
  * every model is tuned with 5-fold cross-validation on the TRAINING rows only
  * the best model is chosen by that cross-validation score, not by the test score
  * the test rows are used once, at the end, to report honest results
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from features import FEATURE_NAMES, MATCH_FEATURES   # noqa: E402
from scoring import baseline_score        # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--features", choices=["all", "match"], default="all")
args = parser.parse_args()
FEATURES = FEATURE_NAMES if args.features == "all" else MATCH_FEATURES
SUFFIX = "" if args.features == "all" else "_" + args.features
print(f"Feature set: {args.features} ({len(FEATURES)} numbers)")

CLASS_NAMES = ["No Fit", "Potential Fit", "Good Fit"]
POINTS = np.array([0, 50, 100])           # 0-100 score = probability-weighted points
MODELS_DIR, REPORTS_DIR = Path("models"), Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load(split):
    df = pd.read_csv(f"data/features_{split}.csv").dropna()
    return df[FEATURES], df["label"].astype(int)


X_train, y_train = load("train")
X_test, y_test = load("test")
print(f"train rows: {len(X_train)} | test rows: {len(X_test)}")
print("train label counts:", y_train.value_counts().sort_index().to_dict(),
      "(0 = No Fit, 1 = Potential Fit, 2 = Good Fit)")
if y_train.nunique() < 3:
    sys.exit("Training data must contain all 3 labels. Run 02_build_features.py on the full train split.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_weights = compute_sample_weight("balanced", y_train)

# name -> (model, settings to try, extra arguments for fit)
candidates = {
    "Logistic Regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        {"logisticregression__C": [0.1, 1, 10]},
        {},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        {"n_estimators": [200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3, 5]},
        {},
    ),
    "XGBoost": (
        XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", subsample=0.8,
                      colsample_bytree=0.8, random_state=42, n_jobs=-1),
        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400]},
        {"sample_weight": train_weights},
    ),
}

results, fitted = [], {}
for name, (model, grid, fit_args) in candidates.items():
    print(f"\nTuning {name} ...")
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=cv, n_jobs=1)
    search.fit(X_train, y_train, **fit_args)
    best = search.best_estimator_
    fitted[name] = best

    proba = best.predict_proba(X_test)
    predicted = proba.argmax(axis=1)
    expected_score = proba @ POINTS
    results.append({
        "model": name,
        "cv_f1_macro": search.best_score_,
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_f1_macro": f1_score(y_test, predicted, average="macro"),
        "test_log_loss": log_loss(y_test, proba, labels=[0, 1, 2]),
        "test_spearman": spearmanr(expected_score, y_test)[0],
        "best_settings": search.best_params_,
    })
    print("  best settings:", search.best_params_)

table = pd.DataFrame(results)

# ---- simple baselines to compare against ----
majority = y_train.value_counts().idxmax()
test_full = pd.read_csv("data/features_test.csv").dropna()   # all 12 numbers, for the baseline
baseline_rows = test_full.apply(
    lambda r: baseline_score(r["tfidf"], r["semantic"],
                             r["skill_coverage"] if r["jd_has_skills"] else None), axis=1)
print("\nBaselines on the test rows:")
print(f"  always predict the most common label: accuracy = {(y_test == majority).mean():.3f}")
print(f"  the Step 5 formula: Spearman correlation with the label = "
      f"{spearmanr(baseline_rows, y_test)[0]:.3f}")

# ---- results table ----
show = table.drop(columns="best_settings").round(3)
print("\n=== RESULTS ===")
print(show.to_string(index=False))
print("\nHow to read it:")
print("  cv_f1_macro   quality on the training rows (used to pick the winner)")
print("  test_*        quality on rows the models never saw during training")
print("  test_spearman how well the 0-100 score orders No Fit < Potential < Good (1 = perfect)")
print("  test_log_loss lower is better (are the probabilities trustworthy?)")
table.to_csv(REPORTS_DIR / f"model_comparison{SUFFIX}.csv", index=False)

# ---- pick the winner by cross-validation score ----
winner = table.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
print(f"\nWinner (best cross-validation F1): {winner}")

# ---- confusion matrices and feature importance ----
importances = {}
for name, model in fitted.items():
    if name == "Logistic Regression":
        importances[name] = np.abs(model[-1].coef_).mean(axis=0)
    else:
        importances[name] = model.feature_importances_

print("\nTop features for each model:")
for name, values in importances.items():
    order = np.argsort(values)[::-1][:5]
    print(f"  {name}: " + ", ".join(f"{FEATURES[i]} ({values[i]:.2f})" for i in order))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, model) in zip(axes, fitted.items()):
        matrix = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1, 2])
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(range(3), CLASS_NAMES, rotation=20)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, matrix[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_matrices{SUFFIX}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, values) in zip(axes, importances.items()):
        order = np.argsort(values)
        ax.barh([FEATURES[i] for i in order], values[order], color="#B45309")
        ax.set_title(f"{name}: feature importance")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"feature_importance{SUFFIX}.png", dpi=150)
    plt.close(fig)
    print("\nCharts saved in the reports folder.")
except ImportError:
    print("\n(matplotlib is not installed, so no charts were made: pip install matplotlib)")

# ---- save the winner ----
joblib.dump({"model": fitted[winner], "name": winner, "features": FEATURES},
            MODELS_DIR / "score_model.joblib")
info = {
    "winner": winner,
    "feature_set": args.features,
    "features": FEATURES,
    "classes": CLASS_NAMES,
    "points": POINTS.tolist(),
    "results": json.loads(table.to_json(orient="records")),
}
(MODELS_DIR / "score_model_info.json").write_text(json.dumps(info, indent=2))
print(f"Saved the winner to {MODELS_DIR / 'score_model.joblib'} (feature set: {args.features})")
