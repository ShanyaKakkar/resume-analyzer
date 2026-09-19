import json
import sys

from resume_data import parse_resume

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/my_resume.pdf"
record = parse_resume(path)
text = record.pop("raw_text")
print(json.dumps(record, indent=2, ensure_ascii=False))
print(f"\n(raw text: {len(text)} characters, not shown)")
