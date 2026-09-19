import re

from skills_extractor import PATTERNS, find_skills

# ---------- Small helpers ----------

# Words that usually start a resume bullet. Used to tell bullets from headings.
ACTION_VERBS = {
    "achieved", "administered", "analysed", "analyzed", "applied", "architected",
    "assisted", "authored", "boosted", "built", "collaborated", "completed",
    "conducted", "configured", "contributed", "coordinated", "created", "debugged",
    "delivered", "deployed", "designed", "developed", "documented", "drove",
    "enhanced", "engineered", "established", "evaluated", "executed", "explored",
    "fixed", "gained", "generated", "handled", "helped", "identified", "implemented",
    "improved", "increased", "initiated", "integrated", "launched", "learned",
    "led", "leveraged", "made", "maintained", "managed", "mentored", "migrated", "monitored",
    "optimised", "optimized", "organised", "organized", "participated", "performed",
    "prepared", "presented", "processed", "published", "ran", "reduced", "refactored",
    "researched", "resolved", "responsible", "reviewed", "secured", "solved",
    "spearheaded", "streamlined", "supported", "taught", "tested", "trained",
    "translated", "used", "utilised", "utilized", "wrote", "worked",
}

BULLET_MARK_RE = re.compile(
    r"^\s*[\u2022\u25cf\u25aa\u25e6\u25cb\u2023\u27a2\u27a4\u25ba\u2013\u2014\-\*\u00b7]+\s*"
)

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
DATE = (r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*)?"
        r"(?:19|20)\d{2}")
DATE_RANGE_RE = re.compile(
    rf"({DATE})\s*(?:-|\u2013|\u2014|to)\s*({DATE}|present|current|ongoing|till date|now)",
    re.I,
)

ROLE_RE = re.compile(
    r"\b(intern|internship|trainee|engineer|developer|analyst|manager|associate|"
    r"consultant|lead|executive|designer|scientist|officer|assistant|architect|"
    r"administrator|coordinator|researcher|tester|specialist|programmer|apprentice|"
    r"volunteer|fellow|head|director|founder|member|representative)\b",
    re.I,
)
COMPANY_SUFFIX_RE = re.compile(r"^(inc|inc\.|ltd|ltd\.|llc|pvt|pvt\.|pvt\. ltd\.?|limited|corp\.?)$", re.I)
TECH_LINE_RE = re.compile(
    r"^(?:tech(?:nologies)?(?:\s+stack|\s+used)?|tools(?:\s+used)?|stack|built with)\s*[:\-]\s*(.*)$",
    re.I,
)
TRAILING_CONNECTORS_RE = re.compile(
    r"(?:\s+(?:using|with|in|on|built with|developed with|made with|by)|[\s\-|:\u2013\u2014(\[,])+$",
    re.I,
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def strip_marker(line):
    return BULLET_MARK_RE.sub("", line).strip()


def normalize_date(text):
    """'Jun 2025' -> '2025-06', '2022' -> '2022', 'ongoing' -> 'Present'."""
    text = text.strip().lower()
    if not re.search(r"(?:19|20)\d{2}", text):
        return "Present"
    year = re.search(r"(?:19|20)\d{2}", text).group(0)
    month = re.match(r"[a-z]{3}", text)
    if month and month.group(0) in MONTHS:
        return f"{year}-{MONTHS[month.group(0)]:02d}"
    return year


def find_dates(line):
    """Return (start, end, line_without_the_dates)."""
    match = DATE_RANGE_RE.search(line)
    if not match:
        return None, None, line.strip()
    rest = (line[:match.start()] + " " + line[match.end():])
    rest = re.sub(r"\s+", " ", rest).strip(" -|,:\u2013\u2014\t")
    return normalize_date(match.group(1)), normalize_date(match.group(2)), rest


def prepare_lines(text):
    """Split into lines, drop empty ones, and glue wrapped lines together.
    A line that starts with a lowercase letter continues the line above it."""
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if lines and line[0].islower() and not BULLET_MARK_RE.match(line):
            lines[-1] = lines[-1] + " " + line
        else:
            lines.append(line)
    return lines


def looks_like_bullet(line):
    if BULLET_MARK_RE.match(line):
        return True
    text = strip_marker(line)
    first_word = re.split(r"\W+", text.lower(), maxsplit=1)[0]
    return (text.endswith(".") or len(text) > 100 or first_word in ACTION_VERBS)


# ---------- Projects (and the fallback for experience) ----------

def split_title_and_tech(header):
    """'Expense Tracker Web App Python, Flask, MySQL'
    -> ('Expense Tracker Web App', ['Python', 'Flask', 'MySQL'])"""
    first = None
    for pattern in PATTERNS.values():
        match = pattern.search(header)
        if match and (first is None or match.start() < first):
            first = match.start()
    if first is None:
        return header.strip(), []
    title = TRAILING_CONNECTORS_RE.sub("", header[:first]).strip()
    if len(title) < 3:                       # the whole line is the title, like "Python Chatbot"
        return header.strip(), []
    return title, find_skills(header[first:])


def parse_titled_entries(text):
    """Entries that start with a short title line, followed by bullets."""
    entries = []
    previous_was_bullet = False
    for line in prepare_lines(text):
        if looks_like_bullet(line):
            if entries:
                entries[-1]["bullets"].append(strip_marker(line))
            previous_was_bullet = True
            continue

        tech_match = TECH_LINE_RE.match(line)
        if entries and tech_match:
            for skill in find_skills(tech_match.group(1)):
                if skill not in entries[-1]["tech"]:
                    entries[-1]["tech"].append(skill)
            previous_was_bullet = False
            continue

        if not entries or previous_was_bullet:
            start, end, rest = find_dates(line)
            title, tech = split_title_and_tech(rest)
            entries.append({"title": title, "tech": tech, "start_date": start,
                            "end_date": end, "bullets": []})
        else:
            entries[-1]["bullets"].append(line)      # a description line under the title
            previous_was_bullet = True
            continue
        previous_was_bullet = False
    return entries


def extract_projects(text):
    return parse_titled_entries(text)


# ---------- Experience ----------

def split_company(text):
    """'Sample Tech Solutions Pvt. Ltd., Remote' -> ('Sample Tech Solutions Pvt. Ltd.', 'Remote')"""
    chunks = [c.strip() for c in re.split(r"\s*[,|]\s*", text) if c.strip()]
    if not chunks:
        return None, None
    company = chunks[0]
    rest = chunks[1:]
    if rest and COMPANY_SUFFIX_RE.match(rest[0]):        # "Google, Inc., Bangalore"
        company += ", " + rest.pop(0)
    return company, (", ".join(rest) or None)


def build_experience(first_line, second_line, start, end, bullets):
    a, b = first_line, second_line
    if b and ROLE_RE.search(b) and not ROLE_RE.search(a):
        role, company_text = b, a
    elif b:
        role, company_text = a, b
    elif ROLE_RE.search(a):
        role, company_text = a, None
    else:
        role, company_text = None, a
    company, location = split_company(company_text) if company_text else (None, None)
    return {"role": role, "company": company, "location": location,
            "start_date": start, "end_date": end,
            "is_current": end == "Present", "bullets": bullets}


def extract_experience(text):
    """Each entry has a date range, like 'Jun 2025 - Aug 2025'. Two layouts work:
      1. 'Role   dates' on one line, then the company line, then bullets
      2. company and role on their own lines, then the dates alone on the next line."""
    lines = prepare_lines(text)
    date_idx = [i for i, line in enumerate(lines)
                if find_dates(line)[0] and not looks_like_bullet(line)]

    if not date_idx:
        # No dates anywhere: fall back to "title line + bullets"
        result = []
        for e in parse_titled_entries(text):
            parts = re.split(r"\s+(?:at|@)\s+", e["title"], maxsplit=1)
            role = parts[0]
            company = parts[1] if len(parts) > 1 else None
            result.append({"role": role, "company": company, "location": None,
                           "start_date": e["start_date"], "end_date": e["end_date"],
                           "is_current": e["end_date"] == "Present", "bullets": e["bullets"]})
        return result

    # Layout 2: for a date-only line, the (up to 2) plain lines above it are its title lines
    pulled = {}
    for n, i in enumerate(date_idx):
        if find_dates(lines[i])[2]:
            continue
        lower = date_idx[n - 1] + 1 if n else 0
        j = i
        while j > lower and i - j < 2 and not looks_like_bullet(lines[j - 1]):
            j -= 1
        pulled[i] = j

    entries = []
    for n, i in enumerate(date_idx):
        start, end, rest = find_dates(lines[i])
        stop = date_idx[n + 1] if n + 1 < len(date_idx) else len(lines)
        stop = pulled.get(stop, stop)          # those lines belong to the next entry
        body = lines[i + 1:stop]

        if rest:                               # layout 1
            titles = [rest]
            if body and not looks_like_bullet(body[0]):
                titles.append(body.pop(0))
        else:                                  # layout 2
            titles = lines[pulled[i]:i]

        first = titles[0] if titles else ""
        second = titles[1] if len(titles) > 1 else None
        entries.append(build_experience(first, second, start, end,
                                        [strip_marker(b) for b in body]))
    return entries


# ---------- Certifications and achievements ----------

def extract_certifications(text):
    """'Python for Everybody Specialization, Coursera, 2024'
    -> name, issuer, year"""
    items = []
    for line in prepare_lines(text):
        line = strip_marker(line)
        year_match = YEAR_RE.search(line)
        year = int(year_match.group(1)) if year_match else None
        without_year = YEAR_RE.sub("", line)
        chunks = [c.strip(" -:\u2013\u2014") for c in re.split(r"\s*[,|]\s*|\s-\s", without_year)]
        chunks = [c for c in chunks if c]
        if not chunks:
            continue
        items.append({"name": chunks[0],
                      "issuer": chunks[1] if len(chunks) > 1 else None,
                      "year": year})
    return items


def extract_achievements(text):
    return [strip_marker(line) for line in prepare_lines(text) if strip_marker(line)]
