import re

from skills_extractor import SINGLE_LETTER, find_skills

# A heading (or a short line) that switches the "mode" of the lines below it
PREFERRED_HEAD = re.compile(
    r"^\W*(?:preferred|nice to have|good to have|bonus|desirable|optional|added advantage|"
    r"it is a plus|a plus)\b", re.I)
REQUIRED_HEAD = re.compile(
    r"^\W*(?:requirements?|required|must have|qualifications?|responsibilities|"
    r"key skills|skills required|what you(?:'ll| will) (?:need|do|bring)|about you)\b", re.I)

STOPWORDS = set("""
a an the and or of to in for on with at by from as is are was were be been being this that these
those it its we you your our their they them he she his her i me my not no yes if then than so
such can could should would will shall may might must do does did done have has had having get
gets got about into over under after before between through during more most less least very
also just only own same other another each every any all both few many much some what which who
whom when where why how here there out up down off again further once
""".split())

JD_FILLER = set("""
experience experiences work working works team teams ability abilities strong skills skill looking
candidate candidates role roles responsibilities responsibility requirements requirement required
preferred knowledge understanding year years good excellent etc including include includes related
join company opportunity position job description plus using use used based across within new help
helps ensure ensures develop develops developing building build builds design designs support
supports intern interns internship internships fresher freshers apply applicants applicant
should must need needs needed like well able least minimum degree bachelor bachelors master
masters computer science engineering field relevant familiarity proficiency proficient basic
hands day days week weeks month months time full part
write writes writing own improve improves small large high quality clean
""".split())


def stem(word):
    word = word.lower()
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def tokens(text):
    return re.findall(r"[a-zA-Z][a-zA-Z+#]{2,}", text.lower())


def extract_keywords(text, top_n=15):
    """Words that the job description repeats (2 or more times) and that are not
    on our skills list. These are less reliable than skills, so they are shown
    separately."""
    counts = {}
    for word in tokens(text):
        if word in STOPWORDS or word in JD_FILLER or find_skills(word):
            continue
        key = stem(word)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, count in ranked if count >= 2][:top_n]


def analyze_jd(text):
    """Find the required and the preferred skills in a job description."""
    required, preferred = [], []
    mode = "required"
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) <= 60:
            if PREFERRED_HEAD.match(line):
                mode = "preferred"
            elif REQUIRED_HEAD.match(line):
                mode = "required"
        found = find_skills(line) + [s for s, p in SINGLE_LETTER.items() if p.search(line)]
        target = preferred if mode == "preferred" else required
        for skill in found:
            if skill not in target:
                target.append(skill)
    preferred = [s for s in preferred if s not in required]   # required wins
    return {
        "required": required,
        "preferred": preferred,
        "keywords": extract_keywords(text),
    }
