import sys

from resume_parser import extract_text

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
text = extract_text(path)
print(text)
print("-----")
print("Characters:", len(text))