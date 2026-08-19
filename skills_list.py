"""
skills_list.py
----------------
One shared list of known technical skills, used by both the synthetic
data generator and the resume skill extractor. Keeping it in one file
means both sides always agree on what counts as a "skill".
"""

KNOWN_SKILLS = [
    # Programming languages
    "python", "java", "c++", "c", "javascript", "typescript", "sql", "r",
    # Data / ML
    "machine learning", "deep learning", "data analysis", "data visualization",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "nlp", "computer vision", "opencv", "xgboost", "statistics",
    # Web / software
    "html", "css", "react", "node.js", "django", "flask", "rest api",
    "git", "docker", "kubernetes", "linux",
    # Cloud / data infra
    "aws", "azure", "gcp", "power bi", "tableau", "excel", "mysql", "mongodb",
    # Soft-skill-adjacent tech process terms
    "agile", "project management", "communication", "leadership",
]

# Job roles used to generate synthetic job postings, each with the
# skills that role realistically asks for.
JOB_ROLES = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau", "statistics", "data visualization"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "sql"],
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "sql", "data visualization"],
    "Backend Developer": ["python", "java", "django", "flask", "rest api", "sql", "docker", "git"],
    "Frontend Developer": ["javascript", "typescript", "react", "html", "css", "git"],
    "DevOps Engineer": ["docker", "kubernetes", "linux", "aws", "azure", "git"],
    "NLP Engineer": ["python", "nlp", "machine learning", "deep learning", "pytorch", "tensorflow"],
    "Computer Vision Engineer": ["python", "computer vision", "opencv", "deep learning", "pytorch", "tensorflow"],
}
