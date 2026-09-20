import sys

from analyzer import analyze

GOOD_MATCH_JD = """Python Developer Intern - Sample Analytics Pvt. Ltd.

About the role
We are looking for a Python Developer Intern to build backend services and small machine learning features.

Requirements
- Bachelor's degree in Computer Science or a related field
- Good knowledge of Python and SQL
- Experience with Flask or Django and REST APIs
- Familiarity with Git and GitHub
- Understanding of data structures and algorithms
- Basic knowledge of machine learning with Pandas and Scikit-learn

Nice to have
- Docker
- AWS
- Unit testing with Pytest

Responsibilities
- Build and test REST API endpoints for our analytics platform
- Clean data and write reusable Python code for our data pipeline
- Write unit tests and documentation for the analytics services
"""

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
print("Loading the language model (the first run downloads about 90 MB, please wait)...")
result = analyze(path, GOOD_MATCH_JD)

print(f"\nOVERALL SCORE: {result['score']} / 100")
for warning in result["warnings"]:
    print("WARNING:", warning)
print("part scores:", result["parts"])
print("weights used:", result["weights_used"])
if result["not_scored"]:
    print("not scored:", ", ".join(result["not_scored"]))

for name, section in result["sections"].items():
    score = section["score"]
    print(f"\n=== {name.upper()}  " + (f"{score}/100" if score is not None else "(not scored)"))
    for item in section["feedback"]:
        print(f"  [{item['type']}] {item['text']}")
        for example in item.get("examples", []):
            print(f"        e.g. {example}")

print("\n=== ROLE FIT (from skill profiles) ===")
for label in ("resume", "job"):
    roles = result["roles"][label]
    print(f"{label}: " + ("; ".join(f"{r['role']} {r['match']:.0%}" for r in roles) or "nothing clear"))
    if roles and label == "resume":
        print("  skills that match the top role:", ", ".join(roles[0]["matched_skills"]))
        if roles[0]["missing_core"]:
            print("  core skills still missing for it:", ", ".join(roles[0]["missing_core"]))
print("resume and job aligned:", result["roles"]["aligned"])

print("\n=== CATEGORY MODEL (trained on general resumes) ===")
categories = result["categories"]
if categories["note"]:
    print(categories["note"])
for label in ("resume", "job"):
    suggestions = categories[label]
    if suggestions:
        print(f"{label}: " + "; ".join(f"{c['category']} {c['probability']:.0%}" for c in suggestions))
print("\nother missing keywords:", ", ".join(result["missing_keywords"]) or "-")
