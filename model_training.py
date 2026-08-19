"""
model_training.py
--------------------
Trains a classifier that predicts whether a candidate is a "Good Fit"
for a job role, based on skill-overlap features (not raw text — that
keeps the model small, fast, and easy to explain).

Run after data_generator.py:  python model_training.py
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLUMNS = [
    "match_count",
    "match_ratio",
    "total_candidate_skills",
    "extra_skills_count",
    "years_experience",
]

MODEL_PATH = os.path.join("models", "fit_model.pkl")


def load_data():
    data_path = os.path.join("data", "resume_matches.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("data/resume_matches.csv not found. Run data_generator.py first.")
    return pd.read_csv(data_path)


def train_model():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df["is_good_fit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, probs)
    print(f"ROC-AUC: {score:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "roc_auc": score,
        }, f)

    print(f"Saved to {MODEL_PATH}")
    return score


if __name__ == "__main__":
    train_model()
