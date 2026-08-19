"""
resume_parser.py
-------------------
Turns an uploaded resume (PDF or plain text) into plain text, then
extracts which known skills appear in it.

Beginner note: this uses simple keyword matching, not a fancy NLP
model. That's a deliberate, honest choice for a student project —
it's easy to explain in an interview, and it's genuinely how a lot
of real Applicant Tracking Systems (ATS) start out.
"""

import re
from pypdf import PdfReader

from skills_list import KNOWN_SKILLS


def extract_text_from_pdf(uploaded_file):
    """
    Takes a Streamlit-uploaded PDF file object and returns its text.
    """
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text_from_txt(uploaded_file):
    """Reads a plain .txt resume upload."""
    return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_text(text):
    """Lowercases and collapses extra whitespace, for reliable matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_skills(text):
    """
    Scans the resume text for every skill in our known skill list.
    Uses word-boundary matching so "r" doesn't match inside "programmer",
    and "c" doesn't match inside "communication", etc.
    """
    cleaned = clean_text(text)
    found_skills = []

    for skill in KNOWN_SKILLS:
        # Escape special regex characters in skills like "c++" or "node.js"
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, cleaned):
            found_skills.append(skill)

    return found_skills


def extract_years_experience(text):
    """
    Very simple heuristic: looks for patterns like "3 years" or
    "5+ years" and returns the largest number found (0 if none).
    A real ATS would do this more robustly — this is a clear,
    explainable starting point.
    """
    cleaned = clean_text(text)
    matches = re.findall(r"(\d+)\+?\s*years?", cleaned)
    numbers = [int(m) for m in matches]
    return max(numbers) if numbers else 0


def parse_resume(uploaded_file):
    """
    Main entry point called by app.py.
    Returns (raw_text, skills_found, years_experience).
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_txt(uploaded_file)

    skills_found = extract_skills(text)
    years = extract_years_experience(text)

    return text, skills_found, years
