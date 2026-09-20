"""Find the Skills part of a resume, and split it into separate skills.

Used both for training on the Kaggle resumes and for resumes uploaded to the website,
so the model always sees the same kind of text: the skills list, nothing else."""
import re

from section_splitter import split_sections

HEADINGS = [
    "Summary", "Highlights", "Accomplishments", "Experience", "Work History",
    "Professional Experience", "Education", "Education and Training", "Skills",
    "Additional Information", "Certifications", "Interests", "Languages", "Affiliations",
    "Professional Affiliations", "Qualifications", "Core Qualifications", "Awards",
    "Personal Information", "Volunteer Experience", "Objective", "Career Overview",
    "Military Experience", "Presentations", "Publications", "Licenses", "References",
]
# accept "Skills", "SKILLS" and other spellings of the same headings
_ALL = HEADINGS + [h.upper() for h in HEADINGS]
HEADING_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, _ALL), key=len, reverse=True))
    + r")(?![A-Za-z])")
SPLIT_RE = re.compile(r"[,;:\u2022\u00b7|\n\t]|\s{2,}|\s[-\u2013\u2014]\s")


def _section_by_headings(text):
    """For resumes that arrive as one long block of text (like the Kaggle ones)."""
    matches = list(HEADING_RE.finditer(text))
    best = ""
    for i, match in enumerate(matches):
        if match.group(1).lower() != "skills":
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[match.end():end][:2500]
        if len(section.split()) > len(best.split()):
            best = section
    return best


def get_skills_text(resume_text):
    """The skills part of a resume, or '' if it has none."""
    section = split_sections(resume_text or "").get("skills", "")
    if len(section.split()) >= 3:
        return section
    return _section_by_headings(resume_text or "")


def skill_phrases(section):
    """'Python, SQL, Team work' -> {'python', 'sql', 'team work'}"""
    found = set()
    for part in SPLIT_RE.split(section or ""):
        phrase = re.sub(r"^(and|or|the)\s+", "", part.strip().lower())
        phrase = phrase.strip(" .:()[]\"'")
        words = phrase.split()
        if 1 <= len(words) <= 4 and len(phrase) <= 40 and re.fullmatch(r"[a-z0-9&+#./ \-]+", phrase):
            found.add(phrase)
    return found
