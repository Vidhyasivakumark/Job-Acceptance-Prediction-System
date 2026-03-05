import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# --------------------------- ------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide"
)

st.title("📊 Job Acceptance Analytics Dashboard")
st.markdown("### Recruitment & Placement Intelligence System")

# ---------------------------------
# Load Data
# ---------------------------------
df = pd.read_csv("Final_data.csv")

# Convert status to numeric if needed
if df['status'].dtype == 'object':
    df['status'] = df['status'].map({
        'Not Placed': 0,
        'Placed': 1
    })

# ---------------------------------
# Load Model
# ---------------------------------
with open("final_job_acceptance_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

X = df[feature_columns]

# Generate Probability
df["accept_prob"] = model.predict_proba(X)[:,1]

# ---------------------------------
# Sidebar Filters
# ---------------------------------
st.sidebar.header("🔎 Filter Options")

company_filter = st.sidebar.multiselect(
    "Select Company Tier",
    options=df['company_tier'].unique(),
    default=df['company_tier'].unique()
)

experience_filter = st.sidebar.multiselect(
    "Select Experience Category",
    options=df['experience_category'].unique(),
    default=df['experience_category'].unique()
)

filtered_df = df[
    (df['company_tier'].isin(company_filter)) &
    (df['experience_category'].isin(experience_filter))
]

# ---------------------------------
# KPI SECTION
# ---------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_candidates = len(filtered_df)
placement_rate = filtered_df['status'].mean() * 100
dropout_rate = 100 - placement_rate
avg_interview = filtered_df['interview_score'].mean()

col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", f"{placement_rate:.2f}")
col3.metric("Dropout Rate (%)", f"{dropout_rate:.2f}")
col4.metric("Avg Interview Score", f"{avg_interview:.2f}")

# ---------------------------------
# CHART 1 - Acceptance by Company Tier
# ---------------------------------
st.subheader("🏢 Acceptance Rate by Company Tier")

tier_data = pd.crosstab(
    filtered_df['company_tier'],
    filtered_df['status'],
    normalize='index'
) * 100

fig1 = px.bar(
    tier_data,
    barmode='group',
    title="Acceptance Percentage by Company Tier",
    labels={"value": "Percentage", "company_tier": "Company Tier"},
    height=400,
    width=500
)

st.plotly_chart(fig1, use_container_width=False)

# ---------------------------------
# CHART 2 - Experience vs Acceptance
# ---------------------------------
st.subheader("👨‍💼 Experience Category vs Acceptance Rate")

exp_data = filtered_df.groupby('experience_category')['status'].mean() * 100

fig2 = px.bar(
    x=exp_data.index,
    y=exp_data.values,
    labels={'x': 'Experience Category', 'y': 'Acceptance Rate (%)'},
    height=400,
    width=500,
    title="Experience Impact on Acceptance"
)

st.plotly_chart(fig2, use_container_width=False)

# ---------------------------------
# CHART 3 - Skills vs Interview Scatter
# ---------------------------------
st.subheader("🎯 Skills Match vs Interview Score")

fig3 = px.scatter(
    filtered_df,
    x='skills_match_percentage',
    y='interview_score',
    color='status',
    title="Skills Match vs Interview Performance",
    opacity=0.6
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------
# CHART 4 - Risk Distribution
# ---------------------------------
st.subheader("⚠ Risk Category Distribution")

def categorize(prob):
    if prob >= 0.75:
        return "High Acceptance"
    elif prob >= 0.40:
        return "Medium Probability"
    else:
        return "High Drop Risk"

filtered_df["risk_category"] = filtered_df["accept_prob"].apply(categorize)

risk_counts = filtered_df["risk_category"].value_counts()

fig4 = px.pie(
    values=risk_counts.values,
    names=risk_counts.index,
    title="Candidate Risk Segmentation",
    height=400,
    width=500
)

st.plotly_chart(fig4, use_container_width=False)

# ---------------------------------
# CHART 5 - Acceptance Probability Distribution
# ---------------------------------
st.subheader("📈 Acceptance Probability Distribution")

fig5 = px.histogram(
    filtered_df,
    x="accept_prob",
    nbins=30,
    height=400,
    width=500,
    title="Distribution of Acceptance Probability"
)

st.plotly_chart(fig5, use_container_width=False)

# ---------------------------------
# FEATURE IMPORTANCE SECTION
# ---------------------------------
st.subheader("🔬 Feature Importance Analysis")

feature_importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(ascending=False)

top_features = feature_importance.head(10)

fig6 = px.bar(
    x=top_features.values,
    y=top_features.index,
    orientation='h',
    title="Top 10 Important Features"
)

st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")
st.markdown("### ✅ Dashboard Developed for HR Strategic Decision Support")