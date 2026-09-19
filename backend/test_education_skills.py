import sys

from education_extractor import extract_education
from resume_parser import extract_text
from section_splitter import split_sections
from skills_extractor import extract_skills

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
text = extract_text(path)
sections = split_sections(text)

print("=== EDUCATION ===")
for entry in extract_education(sections.get("education", "")):
    print(entry)

print()
print("=== SKILLS ===")
skills = extract_skills(sections.get("skills", ""), text)
print(len(skills), "skills:", ", ".join(skills))
