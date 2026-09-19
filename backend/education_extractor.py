import re

# (label, pattern). Checked in this order, so school levels come first.
DEGREE_PATTERNS = [
    ("Class XII", r"\bclass\s*(?:xii|12)\b|\b12th\b|\bsenior secondary\b|\bhigher secondary\b|\bintermediate\b|\bhsc\b"),
    ("Class X", r"\bclass\s*(?:x|10)\b|\b10th\b|\bmatriculation\b|\bssc\b|\bsecondary school\b|\bhigh school\b"),
    ("B.Tech", r"\bb\.?\s?tech\b|\bbachelor of technology\b"),
    ("B.E.", r"(?-i:\bB\.?E\b\.?)|\bbachelor of engineering\b"),
    ("M.Tech", r"\bm\.?\s?tech\b|\bmaster of technology\b"),
    ("MCA", r"\bmca\b|\bmaster of computer applications?\b"),
    ("BCA", r"\bbca\b|\bbachelor of computer applications?\b"),
    ("B.Sc", r"\bb\.?\s?sc\b\.?|\bbachelor of science\b"),
    ("M.Sc", r"\bm\.?\s?sc\b\.?|\bmaster of science\b"),
    ("B.Com", r"\bb\.?\s?com\b|\bbachelor of commerce\b"),
    ("M.Com", r"\bm\.?\s?com\b|\bmaster of commerce\b"),
    ("BBA", r"\bbba\b|\bbachelor of business administration\b"),
    ("MBA", r"\bmba\b|\bmaster of business administration\b"),
    ("Diploma", r"\bdiploma\b|\bpolytechnic diploma\b"),
    ("PhD", r"\bph\.?\s?d\b\.?|\bdoctorate\b"),
]
DEGREE_REGEXES = [(label, re.compile(p, re.I)) for label, p in DEGREE_PATTERNS]
SCHOOL_LEVELS = {"Class X", "Class XII"}

INSTITUTION_RE = re.compile(
    r"\b(college|university|institute|school|academy|polytechnic|vidyalaya|"
    r"vidyapeeth|iit|nit|iiit|bits)\b",
    re.I,
)

MONTH = r"(?:[A-Za-z]{3,9}\.?\s*)?"
YEAR = r"((?:19|20)\d{2})"
END = r"((?:19|20)\d{2}|present|ongoing|current|expected)"
YEAR_RANGE_RE = re.compile(YEAR + r"\s*(?:-|\u2013|\u2014|to)\s*" + MONTH + END, re.I)
YEAR_SHORT_RANGE_RE = re.compile(YEAR + r"\s*(?:-|\u2013|\u2014|to)\s*(\d{2})\b")
SINGLE_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

CGPA_RE = re.compile(
    r"\b(?:CGPA|C\.G\.P\.A|GPA|SGPA)\b\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?)"
    r"(?:\s*/\s*(\d{1,2}(?:\.\d)?))?",
    re.I,
)
CGPA_AFTER_RE = re.compile(
    r"(\d{1,2}\.\d{1,2})\s*(?:/\s*(\d{1,2}(?:\.\d)?))?\s*(?:CGPA|GPA|SGPA)\b", re.I
)
PERCENT_RE = re.compile(r"(\d{2,3}(?:\.\d{1,2})?)\s*%")
BOARD_RE = re.compile(r"\b(CBSE|ICSE|ISC|CISCE|NIOS|State Board|[A-Z]{2,5} Board)\b")


def find_degree(line):
    for label, regex in DEGREE_REGEXES:
        match = regex.search(line)
        if match:
            return label, match.group(0)
    return None, None


def is_institution(line):
    return bool(INSTITUTION_RE.search(line))


def find_years(text):
    """Return (start_year, end_year). End can be 'Present'."""
    match = YEAR_RANGE_RE.search(text)
    if match:
        start, end = int(match.group(1)), match.group(2)
        return start, (int(end) if end.isdigit() else "Present")
    match = YEAR_SHORT_RANGE_RE.search(text)
    if match:
        start = int(match.group(1))
        return start, int(str(start)[:2] + match.group(2))
    match = SINGLE_YEAR_RE.search(text)
    if match:
        return None, int(match.group(1))
    return None, None


def remove_years(text):
    text = YEAR_RANGE_RE.sub("", text)
    text = YEAR_SHORT_RANGE_RE.sub("", text)
    return SINGLE_YEAR_RE.sub("", text)


def remove_scores(text):
    text = CGPA_RE.sub("", text)
    text = CGPA_AFTER_RE.sub("", text)
    return PERCENT_RE.sub("", text)


def find_field(degree_line, degree_text):
    """'B.Tech in Computer Science and Engineering 2022 - 2026'
    -> 'Computer Science and Engineering'"""
    text = remove_scores(remove_years(degree_line)).replace(degree_text, "", 1)
    text = text.lstrip(" -:\u2013\u2014")            # B.Tech - CSE
    first_chunk = re.split(r"[|,(]|\s-\s|\s\u2013\s|\s\u2014\s", text)[0]
    if not first_chunk.strip():
        inside = re.search(r"\(([^)]+)\)", text)     # B.Tech (Computer Science)
        first_chunk = inside.group(1) if inside else ""
    field = re.sub(r"^\W*(?:(?:in|of)\s+)?", "", first_chunk.strip(), flags=re.I)
    field = field.strip(" -:\u2013\u2014")
    return field if 2 < len(field) <= 60 else None


def find_institution(lines):
    for line in lines:
        if not is_institution(line):
            continue
        cleaned = remove_years(line)
        cleaned = remove_scores(cleaned)
        chunks = [c.strip(" -:\u2013\u2014")
                  for c in re.split(r"\s*[|,\u2022]\s*|\s-\s|\s\u2013\s|\s\u2014\s", cleaned)]
        for chunk in chunks:
            if chunk and is_institution(chunk):
                return chunk
    return None


def parse_entry(lines):
    label, degree_text, degree_line = None, None, ""
    for line in lines:
        label, degree_text = find_degree(line)
        if label:
            degree_line = line
            break

    text = "\n".join(lines)
    start_year, end_year = find_years(degree_line) if degree_line else (None, None)
    if end_year is None:
        start_year, end_year = find_years(text)

    cgpa = cgpa_scale = percentage = None
    match = CGPA_RE.search(text) or CGPA_AFTER_RE.search(text)
    if match:
        cgpa = float(match.group(1))
        cgpa_scale = float(match.group(2)) if match.group(2) else 10.0
    match = PERCENT_RE.search(text)
    if match and float(match.group(1)) <= 100:
        percentage = float(match.group(1))

    board = BOARD_RE.search(text)
    return {
        "degree": label,
        "field": None if label in SCHOOL_LEVELS else find_field(degree_line, degree_text),
        "institution": find_institution(lines),
        "start_year": start_year,
        "end_year": end_year,
        "cgpa": cgpa,
        "cgpa_scale": cgpa_scale,
        "percentage": percentage,
        "board": board.group(1) if board else None,
    }


def extract_education(education_text):
    """Split the Education section into one entry per degree, then read each."""
    entries = []       # each entry is a list of lines
    before_first = []  # lines above the first degree (institution-first layout)

    for line in (education_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        label, _ = find_degree(line)
        if label:
            new_entry = [line]
            if not entries:
                new_entry = before_first + new_entry
            else:
                previous = entries[-1]
                # Institution-first layout: the college line sits above its degree line
                if (len(previous) > 1 and is_institution(previous[-1])
                        and not find_degree(previous[-1])[0]
                        and any(is_institution(l) for l in previous[:-1])):
                    new_entry = [previous.pop()] + new_entry
            entries.append(new_entry)
        elif entries:
            entries[-1].append(line)
        else:
            before_first.append(line)

    return [parse_entry(lines) for lines in entries]
