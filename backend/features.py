"""The numbers we give to the machine learning models.
The same function is used for training and for the live website,
so the model always sees numbers built in the same way."""

from jd_parser import analyze_jd, stem, tokens
from scoring import semantic_score, skill_overlap, tfidf_score
from skills_extractor import extract_skills

FEATURE_NAMES = [
    "tfidf",               # same words and word pairs (0-1)
    "semantic",            # similar meaning (0-1)
    "skill_coverage",      # share of the JD's required skills the resume has
    "preferred_coverage",  # share of the JD's preferred skills the resume has
    "jd_has_skills",       # 1 if we found any skills in the JD, else 0
    "n_jd_skills",         # how many skills the JD asks for
    "n_matched_skills",    # how many of them the resume has
    "n_resume_skills",     # how many skills the resume lists
    "keyword_coverage",    # share of the JD's repeated keywords found in the resume
    "resume_words",
    "jd_words",
    "words_ratio",         # resume length divided by JD length
]


# Numbers that describe how well THIS resume fits THIS job. A score for the website
# should move when the resume changes, so the final model uses only these.
MATCH_FEATURES = ["tfidf", "semantic", "skill_coverage", "preferred_coverage",
                  "keyword_coverage"]
# Everything else: lengths and counts that describe only the job description or only
# the resume. Useful for diagnosis, but risky in a live score.
OTHER_FEATURES = [name for name in FEATURE_NAMES if name not in MATCH_FEATURES]


def build_features(resume_text, jd_text, encoder=None):
    jd = analyze_jd(jd_text)
    resume_skills = extract_skills("", resume_text)
    overlap = skill_overlap(resume_skills, jd)

    n_required, n_preferred = len(jd["required"]), len(jd["preferred"])
    pool_coverage = overlap["coverage"]
    preferred_coverage = (len(overlap["matched_preferred"]) / n_preferred) if n_preferred else 0.0

    resume_stems = {stem(w) for w in tokens(resume_text)}
    keywords = jd["keywords"]
    keyword_coverage = (sum(k in resume_stems for k in keywords) / len(keywords)) if keywords else 0.0

    resume_words, jd_words = len(resume_text.split()), len(jd_text.split())
    return {
        "tfidf": tfidf_score(resume_text, jd_text),
        "semantic": semantic_score(resume_text, jd_text, encoder),
        "skill_coverage": 0.0 if pool_coverage is None else pool_coverage,
        "preferred_coverage": preferred_coverage,
        "jd_has_skills": 0 if pool_coverage is None else 1,
        "n_jd_skills": n_required + n_preferred,
        "n_matched_skills": len(overlap["matched_required"]) + len(overlap["matched_preferred"]),
        "n_resume_skills": len(resume_skills),
        "keyword_coverage": keyword_coverage,
        "resume_words": resume_words,
        "jd_words": jd_words,
        "words_ratio": resume_words / max(jd_words, 1),
    }
