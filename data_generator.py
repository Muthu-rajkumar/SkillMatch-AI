"""
data_generator.py
-------------------
Real resumes are personal data, so this generates a realistic
SYNTHETIC dataset of (resume skills, job role, is_good_fit) examples
for the model to learn from — same approach as FraudShield AI's
synthetic transactions.

Run once:  python data_generator.py
"""

import random
import pandas as pd
import os

from skills_list import KNOWN_SKILLS, JOB_ROLES

random.seed(42)


def make_candidate_skills(job_role, is_good_fit):
    """
    Builds one candidate's skill list.
    If is_good_fit=True: mostly overlaps with the role's real requirements.
    If is_good_fit=False: mostly random/unrelated skills (a mismatch).
    """
    required = JOB_ROLES[job_role]

    if is_good_fit:
        # Take most of the required skills, plus 1-2 random extras
        num_required_to_include = random.randint(max(1, len(required) - 2), len(required))
        picked = random.sample(required, num_required_to_include)
        extras = random.sample(KNOWN_SKILLS, random.randint(0, 2))
        skills = list(set(picked + extras))
    else:
        # Mostly unrelated skills, maybe 1 accidental overlap
        unrelated_pool = [s for s in KNOWN_SKILLS if s not in required]
        num_skills = random.randint(3, 7)
        skills = random.sample(unrelated_pool, min(num_skills, len(unrelated_pool)))
        if random.random() < 0.3:
            skills.append(random.choice(required))

    return list(set(skills))


def skill_overlap_features(candidate_skills, job_role):
    """
    Turns (candidate_skills, job_role) into the numeric features the
    ML model actually trains on.
    """
    required = set(JOB_ROLES[job_role])
    candidate = set(candidate_skills)

    matched = candidate & required
    match_count = len(matched)
    match_ratio = match_count / len(required) if required else 0
    extra_skills_count = len(candidate - required)
    years_experience = random.randint(0, 8)  # simulated experience

    return {
        "match_count": match_count,
        "match_ratio": round(match_ratio, 3),
        "total_candidate_skills": len(candidate),
        "extra_skills_count": extra_skills_count,
        "years_experience": years_experience,
    }


def generate_dataset(num_rows=4000):
    rows = []
    job_roles = list(JOB_ROLES.keys())

    for _ in range(num_rows):
        job_role = random.choice(job_roles)
        is_good_fit = random.random() < 0.4  # ~40% good matches, 60% mismatches

        candidate_skills = make_candidate_skills(job_role, is_good_fit)
        features = skill_overlap_features(candidate_skills, job_role)

        row = {
            "job_role": job_role,
            "candidate_skills": ", ".join(sorted(candidate_skills)),
            **features,
            "is_good_fit": int(is_good_fit),
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(num_rows=4000)
    output_path = os.path.join("data", "resume_matches.csv")
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} rows ({df['is_good_fit'].sum()} good fits).")
