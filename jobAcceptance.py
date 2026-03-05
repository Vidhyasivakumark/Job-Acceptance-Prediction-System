import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


# ------------------------------
# Load Saved Model & Features
# ------------------------------
with open("final_job_acceptance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Job Acceptance Prediction System")

st.title("🎯 Job Acceptance Prediction System")
st.write("Predict whether a candidate will accept or reject a job offer.")

st.sidebar.header("Enter Candidate Details")

# ------------------------------
# Input Fields (Customize based on your features)
# ------------------------------

technical_score = st.sidebar.slider("Technical Score", 0, 100, 70)
interview_score = st.sidebar.slider("Interview Score", 0, 100, 70)
expected_ctc_lpa = st.sidebar.number_input("Expected CTC (LPA)", 1.0, 50.0, 6.0)
skills_match_percentage = st.sidebar.slider("Skills Match %", 0, 100, 70)
years_of_experience = st.sidebar.number_input("Years of Experience", 0.0, 20.0, 2.0)
previous_ctc_lpa = st.sidebar.number_input("Previous CTC (LPA)", 0.0, 50.0, 4.0)
aptitude_score = st.sidebar.slider("Aptitude Score", 0, 100, 65)
communication_score = st.sidebar.slider("Communication Score", 0, 100, 65)
notice_period_days = st.sidebar.number_input("Notice Period (Days)", 0, 180, 30)
employment_gap_months = st.sidebar.number_input("Employment Gap (Months)", 0, 60, 0)
certifications_count = st.sidebar.number_input("Certifications Count", 0, 20, 1)
age_years = st.sidebar.number_input("Age", 18, 60, 24)

# ------------------------------
# Create Input DataFrame
# ------------------------------

input_dict = {
    "technical_score": technical_score,
    "interview_score": interview_score,
    "expected_ctc_lpa": expected_ctc_lpa,
    "skills_match_percentage": skills_match_percentage,
    "years_of_experience": years_of_experience,
    "previous_ctc_lpa": previous_ctc_lpa,
    "aptitude_score": aptitude_score,
    "communication_score": communication_score,
    "notice_period_days": notice_period_days,
    "employment_gap_months": employment_gap_months,
    "certifications_count": certifications_count,
    "age_years": age_years
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Acceptance"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Candidate Likely to Accept Offer")
    else:
        st.error("⚠ Candidate Likely to Reject Offer")

    st.write(f"Acceptance Probability: {round(probability*100,2)}%")

    # Risk Category
    if probability >= 0.75:
        st.info("📈 High Acceptance Probability Candidate")
    elif probability >= 0.40:
        st.warning("⚠ Medium Probability Candidate")
    else:
        st.error("🚨 High Drop Risk Candidate")