"""
matcher.py
-----------
Bridges resume_parser.py (what skills a candidate has) with the
trained model (fit_model.pkl) to produce a final Job-Fit Score.
"""

import os
import pickle

from skills_list import JOB_ROLES

MODEL_PATH = os.path.join("models", "fit_model.pkl")


def model_exists():
    return os.path.exists(MODEL_PATH)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_features(candidate_skills, job_role, years_experience):
    """
    Same feature formula used in data_generator.py — the model was
    trained on features shaped exactly like this, so this must match.
    """
    required = set(JOB_ROLES[job_role])
    candidate = set(candidate_skills)

    matched = candidate & required
    match_count = len(matched)
    match_ratio = match_count / len(required) if required else 0
    extra_skills_count = len(candidate - required)

    return {
        "match_count": match_count,
        "match_ratio": round(match_ratio, 3),
        "total_candidate_skills": len(candidate),
        "extra_skills_count": extra_skills_count,
        "years_experience": years_experience,
    }, matched, required - candidate  # also return matched + missing skills for display


def score_candidate(candidate_skills, job_role, years_experience):
    """
    Main function called by app.py.
    Returns (fit_probability, matched_skills, missing_skills).
    """
    import pandas as pd

    saved = load_model()
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    features, matched, missing = build_features(candidate_skills, job_role, years_experience)
    row = pd.DataFrame([features])[feature_columns]

    fit_probability = model.predict_proba(row)[0][1]
    return fit_probability, sorted(matched), sorted(missing)


def rank_candidates(list_of_candidates, job_role):
    """
    Takes a list of (name, skills, years_experience) tuples and
    returns them ranked from best to worst fit for one job role.
    Used by the "rank multiple resumes" dashboard page.
    """
    results = []
    for name, skills, years in list_of_candidates:
        probability, matched, missing = score_candidate(skills, job_role, years)
        results.append({
            "name": name,
            "fit_score": round(probability * 100, 1),
            "matched_skills": ", ".join(matched),
            "missing_skills": ", ".join(missing),
        })

    results.sort(key=lambda r: r["fit_score"], reverse=True)
    return results
