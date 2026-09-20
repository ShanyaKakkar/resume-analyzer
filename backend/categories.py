"""Suggest job categories for a resume (or a job description) with the trained model."""
import json
from pathlib import Path

import numpy as np

from skills_text import get_skills_text, skill_phrases

MODELS_DIR = Path("models")
_cache = {}


def _load():
    if "bundle" not in _cache:
        path = MODELS_DIR / "category_model.joblib"
        if path.exists():
            import joblib
            _cache["bundle"] = joblib.load(path)
        else:
            _cache["bundle"] = None
        typical = MODELS_DIR / "category_typical_skills.json"
        _cache["typical"] = json.loads(typical.read_text()) if typical.exists() else {}
    return _cache["bundle"], _cache["typical"]


def pretty(name):
    return name.replace("-", " ").title()


MIN_CONFIDENCE = 0.50    # below this, the model is guessing, so we show nothing


def model_available():
    return _load()[0] is not None


def suggest_categories(text, top_k=3, min_confidence=MIN_CONFIDENCE):
    """[{'category': 'Information Technology', 'probability': 0.62, 'matched_skills': [...]}, ...]
    Returns an empty list if the model is missing or is not confident enough."""
    bundle, typical = _load()
    if bundle is None or not (text or "").strip():
        return []
    drop = bundle.get("drop_first_words", 0)
    shortened = " ".join(text.split()[drop:])
    probabilities = bundle["model"].predict_proba([shortened])[0]
    if probabilities.max() < min_confidence:
        return []
    resume_skills = skill_phrases(get_skills_text(text))
    result = []
    for index in np.argsort(probabilities)[::-1][:top_k]:
        name = bundle["classes"][index]
        matched = [s for s in typical.get(name, []) if s in resume_skills]
        result.append({"category": pretty(name), "probability": round(float(probabilities[index]), 3),
                       "matched_skills": matched[:8]})
    return result
