"""Download the resume / job description fit dataset and show a summary."""
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

for split in dataset:
    df = dataset[split].to_pandas()
    df.to_csv(OUT / f"fit_{split}.csv", index=False)

    print(f"=== {split}: {len(df)} rows ===")
    print("columns:", list(df.columns))
    print("labels:")
    print(df["label"].value_counts().to_string())
    print("average words per resume:", int(df["resume_text"].str.split().str.len().mean()))
    print("average words per job description:", int(df["job_description_text"].str.split().str.len().mean()))
    print("unique resumes:", df["resume_text"].nunique(),
          "| unique job descriptions:", df["job_description_text"].nunique())
    print()

print("Saved to data/raw/ (this folder is not uploaded to GitHub).")
