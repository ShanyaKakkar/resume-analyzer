"""Save the range of meaning-similarity values seen in real resume/job pairs, so the website
can turn a raw similarity into a 0-100 number in a way the data supports.

  python training/12_make_calibration.py"""
import json
from pathlib import Path

import pandas as pd

features = pd.read_csv("data/features_train.csv")
low, high = features["semantic"].quantile(0.05), features["semantic"].quantile(0.95)
Path("models").mkdir(exist_ok=True)
Path("models/score_calibration.json").write_text(json.dumps(
    {"semantic_low": round(float(low), 3), "semantic_high": round(float(high), 3)}, indent=2))
print(f"5th percentile {low:.3f}, 95th percentile {high:.3f}. Saved models/score_calibration.json")
