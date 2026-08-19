"""
app.py
-------
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os

import data_generator
import model_training
import matcher
import resume_parser
from skills_list import JOB_ROLES, KNOWN_SKILLS

st.set_page_config(page_title="SkillMatch AI", page_icon="🧠", layout="wide")

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8B5CF6, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #A399C4;
        font-size: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: -8px;
    }
    div[data-testid="stMetric"] {
        background-color: #1A1430;
        border: 1px solid #2E2450;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #8B5CF6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<p class="main-header">SkillMatch AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Resume Screening & Job-Fit Prediction</p>', unsafe_allow_html=True)
st.write("")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📄 Screen a Resume",
        "📊 Rank Multiple Resumes",
        "⚙️ Train / Retrain Model",
    ],
)

DATA_PATH = os.path.join("data", "resume_matches.csv")


# ---------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------
if page == "🏠 Dashboard":
    if not os.path.exists(DATA_PATH):
        st.warning("No training data yet. Go to 'Train / Retrain Model' to generate data and train the model.")
    else:
        df = pd.read_csv(DATA_PATH)

        col1, col2, col3 = st.columns(3)
        col1.metric("Training Examples", len(df))
        col2.metric("Good Fit Examples", int(df["is_good_fit"].sum()))
        col3.metric("Job Roles Covered", df["job_role"].nunique())

        st.subheader("Good-Fit Rate by Job Role")
        role_summary = df.groupby("job_role")["is_good_fit"].mean().reset_index()
        role_summary["is_good_fit"] = (role_summary["is_good_fit"] * 100).round(1)
        fig = px.bar(role_summary, x="job_role", y="is_good_fit",
                     color="is_good_fit", color_continuous_scale="Purples",
                     labels={"is_good_fit": "Good Fit %"})
        st.plotly_chart(fig, use_container_width=True)

        if matcher.model_exists():
            saved = matcher.load_model()
            st.info(f"Active model ROC-AUC on test data: **{saved['roc_auc']:.3f}**")


# ---------------------------------------------------------------
# PAGE: SCREEN A SINGLE RESUME
# ---------------------------------------------------------------
elif page == "📄 Screen a Resume":
    st.subheader("Upload a Resume and Check Job Fit")

    if not matcher.model_exists():
        st.warning("No trained model found yet. Go to 'Train / Retrain Model' first.")
    else:
        job_role = st.selectbox("Target Job Role", list(JOB_ROLES.keys()))
        uploaded_file = st.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])

        manual_mode = st.checkbox("Or enter skills manually instead of uploading")
        manual_skills = []
        manual_years = 0
        if manual_mode:
            manual_skills = st.multiselect("Select skills", KNOWN_SKILLS)
            manual_years = st.number_input("Years of experience", min_value=0, max_value=40, value=1)

        if st.button("Check Job Fit"):
            if manual_mode:
                if not manual_skills:
                    st.error("Select at least one skill.")
                else:
                    skills_found, years = manual_skills, manual_years
                    show_results = True
            elif uploaded_file is not None:
                raw_text, skills_found, years = resume_parser.parse_resume(uploaded_file)
                show_results = True
                st.caption(f"Extracted {len(skills_found)} known skills and detected ~{years} years of experience from the resume.")
            else:
                st.error("Upload a resume or switch to manual skill entry.")
                show_results = False

            if 'show_results' in dir() and show_results:
                probability, matched, missing = matcher.score_candidate(skills_found, job_role, years)

                st.metric("Job-Fit Score", f"{probability * 100:.1f}%")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Matched Skills**")
                    st.write(", ".join(matched) if matched else "None found")
                with col_b:
                    st.markdown("**⚠️ Missing Skills**")
                    st.write(", ".join(missing) if missing else "None — full match!")

                if probability >= 0.7:
                    st.success("Strong fit for this role.")
                elif probability >= 0.4:
                    st.warning("Partial fit — some key skills missing.")
                else:
                    st.error("Weak fit for this role based on current skills.")


# ---------------------------------------------------------------
# PAGE: RANK MULTIPLE RESUMES
# ---------------------------------------------------------------
elif page == "📊 Rank Multiple Resumes":
    st.subheader("Rank Multiple Candidates for One Job Role")
    st.caption("Upload several resumes at once — they'll be ranked from best to worst fit.")

    if not matcher.model_exists():
        st.warning("No trained model found yet. Go to 'Train / Retrain Model' first.")
    else:
        job_role = st.selectbox("Target Job Role", list(JOB_ROLES.keys()), key="rank_role")
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Rank Candidates"):
            candidates = []
            for f in uploaded_files:
                _, skills_found, years = resume_parser.parse_resume(f)
                candidates.append((f.name, skills_found, years))

            ranked = matcher.rank_candidates(candidates, job_role)
            ranked_df = pd.DataFrame(ranked)
            ranked_df.insert(0, "Rank", range(1, len(ranked_df) + 1))

            st.dataframe(ranked_df, use_container_width=True)

            csv_download = ranked_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Ranking", csv_download, "ranked_candidates.csv", "text/csv")


# ---------------------------------------------------------------
# PAGE: TRAIN / RETRAIN MODEL
# ---------------------------------------------------------------
elif page == "⚙️ Train / Retrain Model":
    st.subheader("Generate Data & Train Model")
    st.caption(
        "Real resumes are personal data, so this generates a realistic synthetic "
        "dataset of skill-overlap examples across 8 job roles for the model to learn from."
    )

    num_rows = st.slider("Number of synthetic examples to generate", 1000, 10000, 4000, step=500)

    if st.button("1️⃣ Generate Sample Data"):
        df = data_generator.generate_dataset(num_rows=num_rows)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"Generated {len(df)} examples ({int(df['is_good_fit'].sum())} good fits) → data/resume_matches.csv")

    if st.button("2️⃣ Train Model"):
        if not os.path.exists(DATA_PATH):
            st.error("Generate data first (step 1).")
        else:
            with st.spinner("Training model..."):
                score = model_training.train_model()
            st.success(f"Training complete — ROC-AUC: {score:.3f}. Saved to models/fit_model.pkl")
