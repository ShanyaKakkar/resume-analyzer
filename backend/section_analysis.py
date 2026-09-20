"""Look at every part of a resume (skills, experience, projects, education, certifications,
achievements) and compare it with a job description.

Each section gets a relevance score (how well it supports THIS job) and plain feedback.
The overall score is a weighted average of the section scores. The weights are our own
choice, written down below, not something a model learned."""
import json
import re
from datetime import date
from pathlib import Path

import numpy as np

from scoring import semantic_score, skill_overlap
from skills_extractor import find_skills

MODELS_DIR = Path("models")

# Our choice of how much each part counts. Parts that are missing or cannot be judged
# (for example a job description that names no degree) are left out and the rest re-scaled.
WEIGHTS = {"skills": 0.35, "text_similarity": 0.20, "experience": 0.15,
           "projects": 0.15, "education": 0.10, "certifications": 0.05}
# how many of the job's skills a section must show to get 100
EVIDENCE_TARGET = {"experience": 5, "projects": 5, "certifications": 2}

WEAK_OPENERS = {
    "worked on": "Built, Developed or Implemented",
    "worked with": "Collaborated with or Partnered with",
    "responsible for": "Led, Managed or Owned",
    "helped with": "Supported or Contributed to",
    "helped in": "Supported or Contributed to",
    "helped": "Supported or Contributed to",
    "assisted": "Supported or Coordinated",
    "involved in": "Delivered or Executed",
    "participated in": "Contributed to or Organized",
    "took part in": "Contributed to or Organized",
    "handled": "Managed or Resolved",
}

DEGREE_LEVEL = {"Class X": 0, "Class XII": 0, "Diploma": 1, "B.Tech": 2, "B.E.": 2, "BCA": 2,
                "B.Sc": 2, "B.Com": 2, "BBA": 2, "M.Tech": 3, "MCA": 3, "M.Sc": 3,
                "M.Com": 3, "MBA": 3, "PhD": 4}
JD_DEGREE_PATTERNS = [
    (4, r"\bph\.?\s?d\b|\bdoctorate\b"),
    (3, r"\bmaster'?s\b|\bmasters?\s+(?:degree|of)\b|\bmaster of\b|\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmca\b|\bmba\b|\bpost[- ]?graduate\b"),
    (2, r"\bbachelor'?s?\b|\bb\.?\s?tech\b|(?-i:\bB\.?E\b)|\bb\.?\s?sc\b|\bbca\b"),
    (1, r"\bdiploma\b"),
]
GENERIC_DEGREE = r"\bdegree\b|\bgraduates?\b"     # counts as a bachelor's only if nothing else is named
LEVEL_NAME = {1: "a diploma", 2: "a bachelor's degree", 3: "a master's degree", 4: "a PhD"}


# ---------- small helpers ----------

def note(kind, text, examples=None):
    item = {"type": kind, "text": text}
    if examples:
        item["examples"] = examples
    return item


def load_calibration():
    """The range of 'meaning similarity' values seen in real resume/job pairs.
    Made by training/12_make_calibration.py; sensible defaults are used without it."""
    path = MODELS_DIR / "score_calibration.json"
    if path.exists():
        data = json.loads(path.read_text())
        return data["semantic_low"], data["semantic_high"]
    return 0.20, 0.70


def bullet_feedback(bullets):
    """Writing checks that apply to any list of bullet points."""
    bullets = [b for b in bullets if b and b.strip()]
    if not bullets:
        return []
    items = []
    unquantified = [b for b in bullets if not re.search(r"\d", b)]
    if len(unquantified) > len(bullets) / 2:
        items.append(note(
            "issue",
            f"Only {len(bullets) - len(unquantified)} of {len(bullets)} bullets contain a number. "
            "Add measurable results: team size, users, time saved, % improved.",
            unquantified[:2]))

    weak = []
    for bullet in bullets:
        lowered = bullet.lower().lstrip()
        for opener in sorted(WEAK_OPENERS, key=len, reverse=True):
            if lowered.startswith(opener):
                weak.append(f'"{bullet}" -> start with: {WEAK_OPENERS[opener]}')
                break
    if weak:
        items.append(note("issue", "Some bullets start with a weak phrase. Start with a strong "
                          "action verb and say what you achieved.", weak[:3]))

    too_long = [b for b in bullets if len(b.split()) > 35]
    if too_long:
        items.append(note("tip", "Some bullets are very long (over 35 words). Keep each to one "
                          "or two lines.", too_long[:1]))
    if any(re.match(r"^(i|my|we)\b", b.strip().lower()) for b in bullets):
        items.append(note("tip", "Avoid 'I', 'my' and 'we' in bullets. Start with the action."))
    ends = [b.rstrip().endswith(".") for b in bullets]
    if 0 < sum(ends) < len(bullets):
        items.append(note("tip", "Formatting is inconsistent: some bullets end with a full stop "
                          "and some do not. Pick one style."))
    if not items:
        items.append(note("good", "Bullets are specific and start with clear actions."))
    return items


def evidence(section_text, jd_skills, target):
    """Which of the job's skills does this section actually show?"""
    found = [s for s in find_skills(section_text) if s in jd_skills]
    needed = max(1, min(target, len(jd_skills)))
    return found, round(100 * min(1.0, len(found) / needed))


# ---------- one function per section ----------

def analyze_skills(record, jd, shown_elsewhere):
    overlap = skill_overlap(record["skills"], jd)
    required, preferred = jd["required"], jd["preferred"]
    score = None
    if required:
        base = len(overlap["matched_required"]) / len(required)
        if preferred:
            preferred_share = len(overlap["matched_preferred"]) / len(preferred)
            base = 0.75 * base + 0.25 * preferred_share
        score = round(100 * base)
    elif preferred:
        score = round(100 * len(overlap["matched_preferred"]) / len(preferred))

    feedback = []
    if not record["skills"]:
        feedback.append(note("issue", "No skills were found. Add a Skills section with the tools "
                             "and technologies you know."))
    if overlap["missing_required"]:
        feedback.append(note("issue", "The job requires skills your resume does not mention: "
                             + ", ".join(overlap["missing_required"])
                             + ". Add them only if you genuinely have them."))
    if overlap["missing_preferred"]:
        feedback.append(note("tip", "Nice-to-have skills you could add if you know them: "
                             + ", ".join(overlap["missing_preferred"]) + "."))
    listed_only = [s for s in overlap["matched_required"] + overlap["matched_preferred"]
                   if s not in shown_elsewhere]
    if listed_only:
        feedback.append(note("tip", "You list these skills but never show them in a project or "
                             "job. Mention where you used them: " + ", ".join(listed_only) + "."))
    if len(record["skills"]) > 35:
        feedback.append(note("tip", f"You list {len(record['skills'])} skills. A shorter list "
                             "focused on this job reads better."))
    if score is None:
        feedback.append(note("tip", "No known skills were found in the job description, so the "
                             "skills could not be scored."))
    elif not overlap["missing_required"]:
        feedback.append(note("good", "Your resume covers all the required skills."))
    return {"score": score, "matched_required": overlap["matched_required"],
            "missing_required": overlap["missing_required"],
            "matched_preferred": overlap["matched_preferred"],
            "missing_preferred": overlap["missing_preferred"], "feedback": feedback}


def analyze_experience(record, jd_skills):
    entries = record["experience"]
    if not entries:
        return {"score": None, "feedback": [note(
            "tip", "No work experience found. Add internships, freelance or volunteer work if you have any.")]}
    bullets = [b for e in entries for b in e["bullets"]]
    text = " ".join(f"{e['role'] or ''} {e['company'] or ''}" for e in entries) + " " + " ".join(bullets)
    found, score = evidence(text, jd_skills, EVIDENCE_TARGET["experience"]) if jd_skills else ([], None)
    feedback = bullet_feedback(bullets)
    for entry in entries:
        if len(entry["bullets"]) < 2:
            label = entry["role"] or entry["company"] or "One entry"
            feedback.append(note("tip", f"'{label}' has fewer than 2 bullets. Add 2 to 4 that "
                                 "describe what you did and what changed."))
    if found:
        feedback.append(note("good", "Experience shows these job skills: " + ", ".join(found) + "."))
    elif jd_skills:
        feedback.append(note("issue", "None of the job's skills appear in your experience. Describe "
                             "the tools you used in each role."))
    return {"score": score, "skills_shown": found, "feedback": feedback}


def analyze_projects(record, jd_skills):
    entries = record["projects"]
    if not entries:
        return {"score": None, "feedback": [note(
            "issue", "No projects found. Projects are the strongest proof of skill for students and freshers.")]}
    bullets = [b for e in entries for b in e["bullets"]]
    text = " ".join(f"{e['title']} {' '.join(e['tech'])}" for e in entries) + " " + " ".join(bullets)
    found, score = evidence(text, jd_skills, EVIDENCE_TARGET["projects"]) if jd_skills else ([], None)
    feedback = bullet_feedback(bullets)
    for entry in entries:
        if not entry["tech"]:
            feedback.append(note("tip", f"Project '{entry['title']}' does not list the technologies used."))
    if found:
        feedback.append(note("good", "Projects show these job skills: " + ", ".join(found) + "."))
    elif jd_skills:
        feedback.append(note("issue", "None of the job's skills appear in your projects. Add a project "
                             "that uses them."))
    return {"score": score, "skills_shown": found, "feedback": feedback}


def resume_level(education):
    """Highest degree level, and whether it is still in progress."""
    best, in_progress = None, False
    this_year = date.today().year
    for entry in education:
        level = DEGREE_LEVEL.get(entry["degree"])
        if level is None:
            continue
        if best is None or level > best:
            best = level
            end = entry["end_year"]
            in_progress = end == "Present" or (isinstance(end, int) and end >= this_year)
    return best, in_progress


def jd_required_level(jd_text):
    levels = [level for level, pattern in JD_DEGREE_PATTERNS if re.search(pattern, jd_text, re.I)]
    if levels:
        return min(levels)                     # "Bachelor's or Master's" asks for a bachelor's
    return 2 if re.search(GENERIC_DEGREE, jd_text, re.I) else None


def analyze_education(record, jd_text):
    entries = record["education"]
    if not entries:
        return {"score": None, "feedback": [note("issue", "No education details were found. Add your "
                                                 "degree, college and years.")]}
    feedback = []
    level, in_progress = resume_level(entries)
    required = jd_required_level(jd_text)
    score = None
    if required is not None and level is not None:
        if level >= required:
            score = 100
            feedback.append(note("good", f"Your education meets the job's requirement of "
                                 f"{LEVEL_NAME[required]}" + (" (your degree is in progress)." if in_progress else ".")))
        elif level == required - 1:
            score = 50
            feedback.append(note("issue", f"The job asks for {LEVEL_NAME[required]}. Your highest "
                                 "degree is one level below."))
        else:
            score = 0
            feedback.append(note("issue", f"The job asks for {LEVEL_NAME[required]}, which is above "
                                 "your listed education."))
    elif required is None:
        feedback.append(note("tip", "The job description names no degree requirement, so education "
                             "was not scored."))
    elif level is None:
        feedback.append(note("issue", "The job names a degree requirement, but no degree could be "
                             "recognised in your education section. Write it in full, like 'B.Tech in "
                             "Computer Science'."))
    for entry in entries:
        college_level = DEGREE_LEVEL.get(entry["degree"], 0) >= 1
        if college_level and entry["cgpa"] is None and entry["percentage"] is None:
            feedback.append(note("tip", f"Add your CGPA or percentage for {entry['degree']}."))
        if entry["end_year"] is None:
            feedback.append(note("tip", f"Add the years for {entry['degree'] or 'each education entry'}."))
        if entry["institution"] is None:
            feedback.append(note("tip", f"Add the college or school name for {entry['degree'] or 'each entry'}."))
    return {"score": score, "highest_level": level, "feedback": feedback}


def analyze_certifications(record, jd_skills):
    entries = record["certifications"]
    if not entries:
        return {"score": None, "feedback": [note(
            "tip", "No certifications found. A certificate in a skill the job asks for adds credibility.")]}
    text = " ".join(e["name"] for e in entries)
    found, score = evidence(text, jd_skills, EVIDENCE_TARGET["certifications"]) if jd_skills else ([], None)
    feedback = []
    if found:
        feedback.append(note("good", "Certifications relate to these job skills: " + ", ".join(found) + "."))
    elif jd_skills:
        feedback.append(note("tip", "None of your certifications match the job's skills. Consider one "
                             "in a required skill you are missing."))
    for entry in entries:
        if entry["year"] is None:
            feedback.append(note("tip", f"Add the year for '{entry['name']}'."))
    return {"score": score, "skills_shown": found, "feedback": feedback}


def analyze_achievements(record):
    items = record["achievements"]
    if not items:
        return {"score": None, "feedback": [note(
            "tip", "No achievements found. Add 2 or 3: ranks, hackathons, awards, problems solved.")]}
    return {"score": None, "feedback": bullet_feedback(items)}


# ---------- put it together ----------

def analyze_sections(record, resume_text, jd_text, jd, encoder=None, use_semantic=True):
    jd_skills = jd["required"] + jd["preferred"]
    exp_text = " ".join(f"{e['role'] or ''} {' '.join(e['bullets'])}" for e in record["experience"])
    proj_text = " ".join(f"{e['title']} {' '.join(e['tech'])} {' '.join(e['bullets'])}" for e in record["projects"])
    cert_text = " ".join(e["name"] for e in record["certifications"])
    shown_elsewhere = set(find_skills(exp_text + " " + proj_text + " " + cert_text))

    sections = {
        "skills": analyze_skills(record, jd, shown_elsewhere),
        "experience": analyze_experience(record, jd_skills),
        "projects": analyze_projects(record, jd_skills),
        "education": analyze_education(record, jd_text),
        "certifications": analyze_certifications(record, jd_skills),
        "achievements": analyze_achievements(record),
    }

    similarity = None
    if use_semantic:
        raw = semantic_score(resume_text, jd_text, encoder)
        low, high = load_calibration()
        similarity = round(100 * min(1.0, max(0.0, (raw - low) / (high - low))))
    parts = {name: sections[name]["score"] for name in
             ("skills", "experience", "projects", "education", "certifications")}
    parts["text_similarity"] = similarity

    available = {k: v for k, v in parts.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in available)
    overall = round(sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight) if available else 0
    warnings = []
    if len(available) < 3:
        warnings.append(f"Only {len(available)} part(s) could be scored, so treat this score as a rough "
                        "guide. Paste the complete job description, and make sure the resume has "
                        "skills, projects and education.")
    return {
        "score": overall,
        "warnings": warnings,
        "parts": parts,
        "weights_used": {k: round(WEIGHTS[k] / total_weight, 2) for k in available},
        "not_scored": [k for k, v in parts.items() if v is None],
        "sections": sections,
    }
