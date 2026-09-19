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
}

# Flat lookup: alias text -> section name
ALIAS_TO_SECTION = {
    alias: section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


def normalize_line(line):
    text = line.lower().replace("&", " and ")
    text = re.sub(r"[^a-z\s]", " ", text)   # remove digits, colons, dashes...
    return re.sub(r"\s+", " ", text).strip()


def find_section(line):
    """Return the section name if this line is a heading, else None."""
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


def split_sections(text):
    sections = {"header": []}
    current = "header"
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name = find_section(line)
        if name:
            current = name
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items() if lines}