"""Turn every (resume, job description) pair into numbers and save them.

Usage:
  python training/02_build_features.py train            # all rows
  python training/02_build_features.py test --limit 200  # quick trial
If it stops, run the same command again: it continues where it stopped."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import scoring                      # noqa: E402
from features import FEATURE_NAMES, build_features   # noqa: E402

LABELS = {"no fit": 0, "potential fit": 1, "good fit": 2}
BATCH = 500


class CachedEncoder:
    """Remembers the vectors of pieces of text it has already seen."""

    def __init__(self, base, max_items=50000):
        self.base, self.cache, self.max_items = base, {}, max_items

    def encode(self, texts, normalize_embeddings=True):
        if len(self.cache) > self.max_items:
            self.cache.clear()
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            vectors = self.base.encode(missing, normalize_embeddings=True, batch_size=64)
            self.cache.update(zip(missing, vectors))
        return np.array([self.cache[t] for t in texts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only use this many rows")
    args = parser.parse_args()

    source = Path(f"data/raw/fit_{args.split}.csv")
    target = Path(f"data/features_{args.split}.csv")
    df = pd.read_csv(source)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)   # same rows every time

    done = len(pd.read_csv(target)) if target.exists() else 0
    if done:
        print(f"Found {done} finished rows in {target}, continuing from there.")
    rows = df.iloc[done:]

    print("Loading the language model...")
    encoder = CachedEncoder(scoring.get_encoder())

    batch = []
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        label = LABELS.get(str(row["label"]).strip().lower())
        if label is None:
            continue
        features = build_features(str(row["resume_text"]), str(row["job_description_text"]), encoder)
        features["label"] = label
        batch.append(features)
        if len(batch) >= BATCH:
            pd.DataFrame(batch).to_csv(target, mode="a", header=not target.exists(), index=False)
            batch = []
    if batch:
        pd.DataFrame(batch).to_csv(target, mode="a", header=not target.exists(), index=False)

    result = pd.read_csv(target)
    print(f"\nDone. {len(result)} rows saved to {target}")
    print("label counts (0 = No Fit, 1 = Potential Fit, 2 = Good Fit):")
    print(result["label"].value_counts().sort_index().to_string())
    print("\naverage of each number, by label:")
    print(result.groupby("label")[FEATURE_NAMES[:4] + ["keyword_coverage"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
