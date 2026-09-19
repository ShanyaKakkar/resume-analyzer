import sys

from details_extractor import extract_contact_details
from resume_parser import extract_links, extract_text
from section_splitter import split_sections

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
text = extract_text(path)
links = extract_links(path)
sections = split_sections(text)
details = extract_contact_details(sections.get("header", ""), text, links)

for key, value in details.items():
    print(f"{key:10} {value}")
print("hidden links found:", links)