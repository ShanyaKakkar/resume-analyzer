import re
from difflib import get_close_matches

# Different resumes call the same section by different names
SECTION_ALIASES = {
    "summary": [
        "summary", "professional summary", "career summary", "objective",
        "career objective", "profile", "about me",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "educational qualifications", "academics", "qualifications",
        "education and training",
    ],
    "skills": [
        "skills", "technical skills", "key skills", "core competencies",
        "skills summary", "technical proficiency", "skills and tools",
        "technical skills and tools", "areas of expertise",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "internships", "internship experience", "employment history",
        "work history", "industrial experience", "training and internships",
    ],
    "projects": [
        "projects", "academic projects", "personal projects", "key projects",
        "project experience", "major projects",
    ],
    "certifications": [
        "certifications", "certificates", "courses", "licenses",
        "licenses and certifications", "courses and certifications",
    ],
    "achievements": [
        "achievements", "awards", "honors", "accomplishments",
        "awards and achievements", "extra curricular activities",
        "positions of responsibility", "activities",
    ],
    # Sections we recognise only so they do not get mixed into other sections.
    # We will NOT store personal_details in the database.
    "interests": ["interests", "hobbies", "hobbies and interests", "interests and hobbies"],
    "languages": ["languages known", "language proficiency", "spoken languages"],
    "personal_details": ["personal details", "personal information", "personal data"],
    "declaration": ["declaration"],
    "references": ["references"],
    "publications": ["publications", "research papers", "research and publications"],
    "volunteering": ["volunteer experience", "volunteering", "social work"],
}

# Flat lookup: alias text -> section name
ALIAS_TO_SECTION = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

# These words are too general to trust when text follows on the same line.
# Example: "Courses: DSA, OS" inside Education should NOT start a new section.
INLINE_BLOCKLIST = {"profile", "courses", "activities", "qualifications", "honors"}


def normalize_line(line):
    text = line.lower().replace("&", " and ")
    text = re.sub(r"[^a-z\s]", " ", text)   # remove digits, colons, dashes...
    return re.sub(r"\s+", " ", text).strip()


def find_section(line):
    """A line that is only a heading, like 'EDUCATION' or '2. Skills:'.
    Returns the section name, or None."""
    if len(line) > 40:          # headings are short
        return None
    text = normalize_line(line)
    if not text:
        return None
    if text in ALIAS_TO_SECTION:
        return ALIAS_TO_SECTION[text]
    # Allow small typos, like "Skils" or "Projectss"
    close = get_close_matches(text, ALIAS_TO_SECTION.keys(), n=1, cutoff=0.88)
    if close:
        return ALIAS_TO_SECTION[close[0]]
    return None


def find_inline_section(line):
    """A heading with content on the same line, like 'Skills: Python, Java, SQL'.
    Returns (section name, the rest of the line), or None."""
    parts = re.split(r"\s*[:\u2013\u2014]\s*|\s+-\s+", line, maxsplit=1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if not right or len(left) > 30:
        return None
    text = normalize_line(left)
    # exact match only here, to avoid false alarms on normal lines
    if text in ALIAS_TO_SECTION and text not in INLINE_BLOCKLIST:
        return ALIAS_TO_SECTION[text], right
    return None


def looks_like_unknown_heading(line):
    """A short ALL CAPS line that is not in our list, like 'HOBBIES'."""
    text = line.strip()
    if len(text) < 4 or len(text) > 30 or len(text.split()) > 4:
        return False
    if not re.fullmatch(r"[A-Za-z&/ ]+", text):
        return False
    return text.isupper()


def split_sections(text):
    sections = {"header": []}
    current = "header"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # 1. A line that is only a known heading
        name = find_section(line)
        if name:
            current = name
            sections.setdefault(current, [])
            continue

        # 2. A known heading with content after it: "Skills: Python, Java"
        inline = find_inline_section(line)
        if inline:
            current, rest = inline
            sections.setdefault(current, []).append(rest)
            continue

        # 3. An unlisted heading: start its own section, so it does not
        #    get mixed into the section above it
        if current != "header" and looks_like_unknown_heading(line):
            current = normalize_line(line).replace(" ", "_")
            sections.setdefault(current, [])
            continue

        sections[current].append(line)

    return {name: "\n".join(lines) for name, lines in sections.items() if lines}