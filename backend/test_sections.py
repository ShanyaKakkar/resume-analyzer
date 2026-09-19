import sys

from resume_parser import extract_text
from section_splitter import find_section, split_sections

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
sections = split_sections(extract_text(path))

for name, content in sections.items():
    print(f"=== {name.upper()} ({len(content.splitlines())} lines) ===")
    print(content[:200])
    print()

print("--- heading variations ---")
for heading in ["WORK EXPERIENCE", "Technical Skills:", "2. Academic Projects", "Skils", "Python"]:
    print(repr(heading), "->", find_section(heading))