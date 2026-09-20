from categories import MIN_CONFIDENCE, model_available, suggest_categories
from jd_parser import analyze_jd
from resume_data import parse_resume
from roles import suggest_roles
from scoring import missing_keywords, semantic_score, tfidf_score
from section_analysis import analyze_sections


def analyze(resume_path, jd_text, encoder=None, use_semantic=True):
    """Compare one resume with one job description, section by section."""
    if not jd_text or len(jd_text.strip()) < 30:
        raise ValueError("Please paste a longer job description.")

    record = parse_resume(resume_path)
    resume_text = record.pop("raw_text")
    jd = analyze_jd(jd_text)

    result = analyze_sections(record, resume_text, jd_text, jd, encoder, use_semantic)

    # 1. Role fit from skill profiles (rule-based, explains itself)
    resume_roles = suggest_roles(record["skills"])
    job_roles = suggest_roles(jd["required"] + jd["preferred"])
    roles_aligned = None
    if resume_roles and job_roles:
        roles_aligned = job_roles[0]["role"] in [r["role"] for r in resume_roles]

    # 2. Broad category from the trained model (only shown when it is confident)
    resume_categories = suggest_categories(resume_text)
    job_categories = suggest_categories(jd_text)
    aligned = None
    if resume_categories and job_categories:
        aligned = job_categories[0]["category"] in [c["category"] for c in resume_categories]
    category_note = None
    if model_available() and not resume_categories:
        category_note = (f"The category model was not confident (below {MIN_CONFIDENCE:.0%}) about this "
                         "resume, so no category is shown. It was trained on general US resumes and "
                         "is not designed for student software resumes.")

    return {
        "score": result["score"],
        "warnings": result["warnings"],
        "parts": result["parts"],
        "weights_used": result["weights_used"],
        "not_scored": result["not_scored"],
        "sections": result["sections"],
        "missing_keywords": missing_keywords(resume_text, jd["keywords"]),
        "roles": {"resume": resume_roles, "job": job_roles, "aligned": roles_aligned},
        "categories": {
            "note": category_note,
            "resume": resume_categories,
            # the model was trained on resumes, so for a job description this is only a hint
            "job": job_categories,
            "aligned": aligned,
        },
        "resume": record,
    }
