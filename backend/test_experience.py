import json
import sys

from experience_extractor import (extract_achievements, extract_certifications,
                                  extract_experience, extract_projects)
from resume_parser import extract_text
from section_splitter import split_sections

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
sections = split_sections(extract_text(path))

def show(title, data):
    print(f"=== {title} ({len(data)}) ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

show("EXPERIENCE", extract_experience(sections.get("experience", "")))
show("PROJECTS", extract_projects(sections.get("projects", "")))
show("CERTIFICATIONS", extract_certifications(sections.get("certifications", "")))
show("ACHIEVEMENTS", extract_achievements(sections.get("achievements", "")))
