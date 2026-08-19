# 🧠 SkillMatch AI
**AI-Powered Resume Screening & Job-Fit Prediction Platform**

> Upload a resume, pick a target job role, and get an instant AI-driven fit score with matched/missing skills.

---

## 🚨 Problem

Recruiters manually skim hundreds of resumes per role, and candidates rarely get clear feedback on *why* they weren't shortlisted. SkillMatch AI screens resumes automatically and explains the score.

---

## ✅ Features

- **Resume Parsing** — extracts text and skills from uploaded PDF/TXT resumes
- **Skill Extraction** — keyword-based matching against a curated technical skill list
- **Job-Fit Score** — ML model (Random Forest) trained on skill-overlap patterns across 8 job roles
- **Matched vs Missing Skills** — clear, explainable breakdown for each candidate
- **Multi-Resume Ranking** — upload several resumes and rank them best-to-worst for one role
- **Analytics Dashboard** — good-fit rate by job role

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Machine Learning | scikit-learn (Random Forest) |
| Resume Parsing | pypdf |
| Data | pandas, numpy |
| Visualization | Plotly |

---

## 📂 Project Structure

```
SkillMatch-AI/
├── data/
│   └── resume_matches.csv      # created by data_generator.py
├── models/
│   └── fit_model.pkl           # created by model_training.py
├── screenshots/
├── .streamlit/
│   └── config.toml             # app theme
├── app.py                      # Streamlit dashboard + navigation
├── skills_list.py              # shared skill list + job role definitions
├── data_generator.py           # builds synthetic skill-overlap training data
├── resume_parser.py            # extracts text/skills from uploaded resumes
├── model_training.py           # trains the job-fit classifier
├── matcher.py                  # scores/ranks candidates against a job role
├── requirements.txt
└── .gitignore
```

---

## 📊 About the Data

Real resumes are personal data, so `data_generator.py` builds a **synthetic dataset** of skill-overlap examples across 8 job roles (Data Analyst, ML Engineer, Data Scientist, Backend/Frontend Developer, DevOps, NLP Engineer, Computer Vision Engineer). This is a standard, honest approach for a student ML project — be upfront that training data is synthetic if asked in a demo/interview.

The model itself doesn't read raw resume text — it trains on **numeric skill-overlap features** (match count, match ratio, extra skills, years of experience), which keeps it fast, small, and easy to explain.

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/SkillMatch-AI.git
cd SkillMatch-AI

pip install -r requirements.txt

streamlit run app.py
```

Then in the app:
1. Go to **⚙️ Train / Retrain Model**
2. Click **"1️⃣ Generate Sample Data"**
3. Click **"2️⃣ Train Model"**
4. Go to **📄 Screen a Resume** — upload a PDF/TXT resume, or check "enter skills manually" to test without a file

No compiled dependencies (no dlib, no TensorFlow) — installs cleanly with a plain `pip install -r requirements.txt` on any system.

---

## 📄 Resume Description

> **SkillMatch AI – AI-Powered Resume Screening Platform**
> Built an ML-based resume screening system that parses PDF/text resumes, extracts technical skills, and predicts job-fit using a Random Forest classifier trained on skill-overlap features across 8 job roles. Designed a Streamlit dashboard for single-resume scoring and multi-candidate ranking with exportable results.
> *Technologies: Python, scikit-learn, pypdf, pandas, Streamlit, Plotly.*

---

## 🗺️ Roadmap

- [ ] TF-IDF/embedding-based skill extraction instead of keyword matching
- [ ] Support for DOCX resumes
- [ ] Auto-suggest missing skills as learning resources
