"""Find the Kaggle Resume Dataset on this computer and show a summary.

  python training/07_check_category_data.py

It looks in data/raw first. If the dataset is still a zip file in your Downloads folder,
it unpacks the zip into data/raw/kaggle_resumes for you. It never prints resume text."""
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
TARGET = RAW / "kaggle_resumes"


def find_csv():
    for path in sorted(RAW.rglob("*.csv")):
        if path.name.startswith(("fit_", "features_")):
            continue
        try:
            columns = pd.read_csv(path, nrows=1).columns
        except Exception:
            continue
        if "Category" in columns and "Resume_str" in columns:
            return path
    return None


def find_zip():
    places = [Path.home() / "Downloads", Path.home() / "OneDrive" / "Downloads", RAW]
    for folder in places:
        if not folder.exists():
            continue
        for candidate in folder.glob("*.zip"):
            try:
                names = zipfile.ZipFile(candidate).namelist()
            except Exception:
                continue
            if any(n.lower().endswith("resume.csv") for n in names):
                return candidate
    return None


csv_path = find_csv()
if csv_path is None:
    archive = find_zip()
    if archive is not None:
        print(f"Found {archive}. Unpacking into {TARGET} (about a minute) ...")
        TARGET.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(archive).extractall(TARGET)
        csv_path = find_csv()

if csv_path is None:
    sys.exit(
        "I could not find the dataset.\n"
        "1. Open https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset\n"
        "2. Click Download, then 'Download dataset as zip'\n"
        "3. Run this script again (it looks in your Downloads folder)."
    )

df = pd.read_csv(csv_path)
print("CSV file:", csv_path)
print("rows:", len(df))
print("columns:", df.columns.tolist())
print("\nresumes per category:")
print(df["Category"].value_counts().to_string())
words = df["Resume_str"].astype(str).str.split().str.len()
print(f"\nwords per resume: average {int(words.mean())}, shortest {words.min()}, longest {words.max()}")
print("exact duplicate resumes:", int(df["Resume_str"].duplicated().sum()))
print("empty resumes:", int((words < 20).sum()))
