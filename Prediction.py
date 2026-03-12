# -------------------------------------------------
# Job Acceptance Prediction System
# Dashboard + Prediction
# -------------------------------------------------

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------------------------
# Page Config
# -------------------------------------------------

st.set_page_config(page_title="Job Acceptance Prediction", layout="wide")

st.title("Job Acceptance Prediction System")

# -------------------------------------------------
# Load Data and Model
# -------------------------------------------------

data = pd.read_csv("Final_data.csv")

model = pickle.load(open("final_job_acceptance_model.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -------------------------------------------------
# Tabs
# -------------------------------------------------

dashboard_tab, prediction_tab = st.tabs(["Dashboard", "Prediction"])

# =================================================
# DASHBOARD
# =================================================

with dashboard_tab:

    st.header("Dataset Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    total_candidates = len(data)
    accepted = data["status"].sum()
    rejected = total_candidates - accepted
    accept_rate = round((accepted / total_candidates) * 100, 2)

    col1.metric("Total Candidates", total_candidates)
    col2.metric("Accepted", accepted)
    col3.metric("Rejected", rejected)
    col4.metric("Acceptance Rate %", accept_rate)

    st.divider()

    colA, colB = st.columns(2)

    with colA:

        fig1 = px.pie(
            data,
            names="status",
            title="Acceptance vs Rejection Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colB:

        fig2 = px.histogram(
            data,
            x="technical_score",
            color="status",
            title="Technical Score Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)

    with colC:

        fig3 = px.box(
            data,
            x="status",
            y="communication_score",
            title="Communication Score vs Decision"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with colD:

        fig4 = px.scatter(
            data,
            x="years_of_experience",
            y="expected_ctc_lpa",
            color="status",
            title="Experience vs Expected Salary"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Academic Performance")

    fig5 = px.scatter(
        data,
        x="degree_percentage",
        y="technical_score",
        color="status",
        title="Degree % vs Technical Score"
    )

    st.plotly_chart(fig5, use_container_width=True)

# =================================================
# PREDICTION
# =================================================

with prediction_tab:

    st.header("Candidate Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.number_input("Age", 18, 60, 25)
        ssc_percentage = st.number_input("SSC Percentage", 0.0, 100.0, 70.0)
        hsc_percentage = st.number_input("HSC Percentage", 0.0, 100.0, 70.0)
        degree_percentage = st.number_input("Degree Percentage", 0.0, 100.0, 70.0)
        degree_specialization = st.selectbox(
            "Degree Specialization",
            ["Information Technology","Computer Science","Mechanical","Electronics"]
        )

    with col2:
        technical_score = st.number_input("Technical Score", 0.0, 100.0, 75.0)
        aptitude_score = st.number_input("Aptitude Score", 0.0, 100.0, 75.0)
        communication_score = st.number_input("Communication Score", 0.0, 100.0, 75.0)
        skills_match_percentage = st.number_input("Skills Match %", 0.0, 100.0, 80.0)
        certifications_count = st.number_input("Certifications", 0, 20, 1)

    with col3:
        internship_experience = st.selectbox("Internship Experience", ["Yes","No"])
        years_of_experience = st.number_input("Years of Experience", 0, 20, 1)
        career_switch_willingness = st.selectbox("Career Switch Willingness", ["Willing","Not Willing"])
        relevant_experience = st.selectbox("Relevant Experience", ["Relevant","Not Relevant"])

    st.divider()

    col4, col5, col6 = st.columns(3)

    with col4:
        previous_ctc_lpa = st.number_input("Previous CTC (LPA)", 0.0, 50.0, 3.0)
        expected_ctc_lpa = st.number_input("Expected CTC (LPA)", 0.0, 50.0, 5.0)

    with col5:
        company_tier = st.selectbox("Company Tier", ["Tier 1","Tier 2","Tier 3"])
        job_role_match = st.selectbox("Job Role Match", ["Matched","Not Matched"])
        competition_level = st.selectbox("Competition Level", ["Low","Medium","High"])

    with col6:
        bond_requirement = st.selectbox("Bond Requirement", ["Required","Not Required"])
        notice_period_days = st.number_input("Notice Period Days", 0, 180, 30)
        layoff_history = st.selectbox("Layoff History", ["Yes","No"])

    employment_gap_months = st.number_input("Employment Gap Months", 0, 60, 0)
    relocation_willingness = st.selectbox("Relocation Willingness", ["Willing","Not Willing"])

    # Default engineered features
    experience_category = "Junior"
    academic_band = "High"
    skills_level = "Medium"
    interview_score = 70
    degree_norm = 0.8
    skills_norm = 0.6
    interview_norm = 0.6
    experience_norm = 0.2

    # -------------------------------------------------
    # Input Dictionary
    # -------------------------------------------------

    input_dict = {
        "age_years":age_years,
        "ssc_percentage":ssc_percentage,
        "hsc_percentage":hsc_percentage,
        "degree_percentage":degree_percentage,
        "degree_specialization":degree_specialization,
        "technical_score":technical_score,
        "aptitude_score":aptitude_score,
        "communication_score":communication_score,
        "skills_match_percentage":skills_match_percentage,
        "certifications_count":certifications_count,
        "internship_experience":internship_experience,
        "years_of_experience":years_of_experience,
        "career_switch_willingness":career_switch_willingness,
        "relevant_experience":relevant_experience,
        "previous_ctc_lpa":previous_ctc_lpa,
        "expected_ctc_lpa":expected_ctc_lpa,
        "company_tier":company_tier,
        "job_role_match":job_role_match,
        "competition_level":competition_level,
        "bond_requirement":bond_requirement,
        "notice_period_days":notice_period_days,
        "layoff_history":layoff_history,
        "employment_gap_months":employment_gap_months,
        "relocation_willingness":relocation_willingness,
        "experience_category":experience_category,
        "academic_band":academic_band,
        "skills_level":skills_level,
        "interview_score":interview_score,
        "degree_norm":degree_norm,
        "skills_norm":skills_norm,
        "interview_norm":interview_norm,
        "experience_norm":experience_norm
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    input_df['gender'] = input_df['gender'].map({
    'male': 1,
    'female': 0
    })

    input_df['job_role_match'] = input_df['job_role_match'].map({
        'matched': 1,
        'not matched': 0
    })

    input_df['relocation_willingness'] = input_df['relocation_willingness'].map({
        'willing': 1,
        'not willing': 0
    })

    input_df['career_switch_willingness'] = input_df['career_switch_willingness'].map({
        'willing': 1,
        'not willing': 0
    })

    input_df['degree_specialization'] = input_df['degree_specialization'].map({
        'Information Technology':3,
        'Computer Science':0,
        'Electronics':1,
        'Mechanical':2,
        'Others':4
    })

    input_df['internship_experience'] = input_df['internship_experience'].map({
        'yes':0,
        'no':1
    })

    input_df['relevant_experience'] = input_df['relevant_experience'].map({
        'Relevant':1,
        'Not Relevant':0
    })

    input_df['layoff_history'] = input_df['layoff_history'].map({
        'Yes':0,
        'No':1
    })

    input_df['company_tier'] = input_df['company_tier'].map({
        'Tier 3':1,
        'Tier 1':3,
        'Tier 2':2
    })

    input_df['competition_level'] = input_df['competition_level'].map({
        'Low':1,
        'Medium':2,
        'High':3
    })

    input_df['bond_requirement'] = input_df['bond_requirement'].map({
        'not required':0,
        'required':1
    })

    input_df['experience_category'] = input_df['experience_category'].map({
        'Fresher':0,
        'Junior':1,
        'Senior':2
    })

    input_df['academic_band'] = input_df['academic_band'].map({
        'Low':0,
        'Medium':1,
        'High':2
    })

    input_df['skills_level'] = input_df['skills_level'].map({
        'Low':0,
        'Medium':1,
        'High':2
    })




    

    # st.subheader("Model Input Data")
    # st.write(input_df)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    if st.button("Predict Candidate Decision"):

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        reject_prob = prob[0]
        accept_prob = prob[1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Candidate is likely to ACCEPT the job offer")
        else:
            st.error("Candidate is likely to REJECT the job offer")

        st.subheader("Probability")

        st.write("Reject Probability:", round(reject_prob,2))
        st.write("Accept Probability:", round(accept_prob,2))

        st.progress(int(accept_prob*100))