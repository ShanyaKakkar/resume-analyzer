import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_encoder():
    """Load the language model once and reuse it. The first run downloads it."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------- 1. Keyword score (TF-IDF) ----------

def tfidf_score(resume_text, jd_text):
    """0 to 1: how many of the same words and word pairs both texts use."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([resume_text, jd_text])
    except ValueError:                       # nothing usable in the text
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


# ---------- 2. Meaning score (sentence-transformers) ----------

def make_chunks(text, min_words=12, max_words=60):
    """The model reads only about 200 words at once, so we cut the text into
    small pieces (a few lines each) and compare piece with piece."""
    chunks, buffer = [], []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        buffer.extend(words)
        if len(buffer) >= min_words:
            for i in range(0, len(buffer), max_words):
                chunks.append(" ".join(buffer[i:i + max_words]))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


MAX_CHUNKS = 48     # about 2,900 words per text; keeps long documents fast


def semantic_score(resume_text, jd_text, encoder=None):
    """0 to 1: for each piece of the job description, find the most similar
    piece of the resume, then average. It understands that "ML" and
    "machine learning" mean the same thing."""
    resume_chunks = make_chunks(resume_text)[:MAX_CHUNKS]
    jd_chunks = make_chunks(jd_text)[:MAX_CHUNKS]
    if not resume_chunks or not jd_chunks:
        return 0.0
    encoder = encoder or get_encoder()
    resume_vectors = encoder.encode(resume_chunks, normalize_embeddings=True)
    jd_vectors = encoder.encode(jd_chunks, normalize_embeddings=True)
    similarity = jd_vectors @ resume_vectors.T          # JD pieces x resume pieces
    return float(np.clip(similarity.max(axis=1).mean(), 0.0, 1.0))


# ---------- 3. Skill overlap ----------

def skill_overlap(resume_skills, jd_info):
    have = {s.lower() for s in resume_skills}

    def split(skills):
        matched = [s for s in skills if s.lower() in have]
        missing = [s for s in skills if s.lower() not in have]
        return matched, missing

    matched_req, missing_req = split(jd_info["required"])
    matched_pref, missing_pref = split(jd_info["preferred"])

    pool = jd_info["required"] or jd_info["preferred"]
    matched_pool = matched_req if jd_info["required"] else matched_pref
    coverage = len(matched_pool) / len(pool) if pool else None

    return {
        "coverage": coverage,                  # None when the JD has no known skills
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_preferred": matched_pref,
        "missing_preferred": missing_pref,
    }


# ---------- Missing keywords ----------

def missing_keywords(resume_text, jd_keywords):
    from jd_parser import stem, tokens
    resume_words = {stem(w) for w in tokens(resume_text)}
    return [k for k in jd_keywords if k not in resume_words]


# ---------- Temporary score (Step 6 replaces this with a trained model) ----------

WEIGHTS = {"skills": 0.40, "semantic": 0.35, "keywords": 0.25}


def baseline_score(tfidf, semantic, skill_coverage):
    """Rough 0-100 score. The stretching numbers below are a first guess,
    because raw similarity values are usually small (0.2 to 0.6)."""
    parts = {"keywords": min(1.0, tfidf / 0.5)}
    if semantic is not None:
        parts["semantic"] = min(1.0, max(0.0, (semantic - 0.2) / 0.5))
    if skill_coverage is not None:
        parts["skills"] = skill_coverage
    total_weight = sum(WEIGHTS[k] for k in parts)
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_weight
    return round(100 * score)
