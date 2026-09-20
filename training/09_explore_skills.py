"""Look at the Skills sections of the Kaggle resumes. Only skill phrases are printed,
never other text from the resumes.

  python training/09_explore_skills.py

We need to know: how many resumes have a Skills section, what the skills look like, and
which skills are typical for each category. This decides how we build the
"skills -> category" step."""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HEADINGS = [
    "Summary", "Highlights", "Accomplishments", "Experience", "Work History",
    "Professional Experience", "Education", "Education and Training", "Skills",
    "Additional Information", "Certifications", "Interests", "Languages", "Affiliations",
    "Professional Affiliations", "Qualifications", "Core Qualifications", "Awards",
    "Personal Information", "Volunteer Experience", "Objective", "Career Overview",
    "Military Experience", "Presentations", "Publications", "Licenses", "References",
]
HEADING_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, HEADINGS), key=len, reverse=True))
    + r")(?![A-Za-z])")
SPLIT_RE = re.compile(r"[,;\u2022\u00b7|\n\t]|\s{2,}|\s[-\u2013\u2014]\s")
FOCUS = ["INFORMATION-TECHNOLOGY", "CHEF", "TEACHER", "ACCOUNTANT", "HEALTHCARE"]


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
    sys.exit("Could not find Resume.csv. Run training/07_check_category_data.py first.")


def skills_section(text):
    """The text under the 'Skills' heading (the longest one, if there are several)."""
    matches = list(HEADING_RE.finditer(text))
    best = ""
    for i, match in enumerate(matches):
        if match.group(1) != "Skills":
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[match.end():end][:2500]
        if len(section.split()) > len(best.split()):
            best = section
    return best


def phrases(section):
    found = set()
    for part in SPLIT_RE.split(section):
        phrase = re.sub(r"^(and|or|the)\s+", "", part.strip().lower())
        phrase = phrase.strip(" .:()[]\"'")
        words = phrase.split()
        if 1 <= len(words) <= 4 and len(phrase) <= 40 and re.fullmatch(r"[a-z0-9&+#./ \-]+", phrase):
            found.add(phrase)
    return found


df = pd.read_csv(find_csv())[["Resume_str", "Category"]].dropna()
df["section"] = df["Resume_str"].astype(str).map(skills_section)
df["phrases"] = df["section"].map(phrases)
has = df["phrases"].map(len) >= 3

print(f"resumes: {len(df)}")
print(f"with a readable Skills section (3 or more skills): {int(has.sum())} ({has.mean():.0%})")
print(f"skills per section: median {int(df.loc[has, 'phrases'].map(len).median())}, "
      f"largest {int(df.loc[has, 'phrases'].map(len).max())}")

overall = Counter(p for s in df.loc[has, "phrases"] for p in s)
print("\nthe 25 most common skills in all resumes:")
print(", ".join(f"{p} ({c})" for p, c in overall.most_common(25)))

rows = []
for category, group in df[has].groupby("Category"):
    counts = Counter(p for s in group["phrases"] for p in s)
    for phrase, count in counts.items():
        share_in = count / len(group)
        share_all = overall[phrase] / has.sum()
        rows.append({"category": category, "skill": phrase, "resumes": count,
                     "share_in_category": share_in, "lift": share_in / share_all})
table = pd.DataFrame(rows)

for category in FOCUS:
    part = table[(table["category"] == category) & (table["resumes"] >= 5)]
    if part.empty:
        continue
    print(f"\n{category}: most common skills")
    print("  " + ", ".join(part.sort_values("resumes", ascending=False).head(10)["skill"]))
    print(f"{category}: most typical skills (much more common here than elsewhere)")
    print("  " + ", ".join(part.sort_values(["lift", "resumes"], ascending=False).head(10)["skill"]))

Path("data/derived").mkdir(parents=True, exist_ok=True)
table.to_csv("data/derived/category_skill_phrases.csv", index=False)
print("\nSaved data/derived/category_skill_phrases.csv (skill phrases only, no resume text).")
