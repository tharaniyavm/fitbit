import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Fitbit Fitness Analytics Hub",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTitle {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #00B2B2; /* Fitbit-style Teal */
        font-weight: 700;
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 15px;
        background-color: #e1f5f5;
        color: #007a7a;
        font-weight: 500;
        font-size: 0.9rem;
        border: 1px solid #00B2B2;
    }
    .tech-badge {
        background-color: #2c3e50;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        margin-right: 5px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.title("🔥 Fitbit: Calorie Burn & Workout Clustering")
st.subheader("Leveraging Machine Learning for Personalized Health Insights")

st.markdown("---")

# Project Summary & Domain
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Problem Statement")
    st.info("""
    Accurate calorie estimation is vital for modern fitness apps. Beyond heart rate and duration, 
    this project integrates factors like workout type, hydration, and user experience to:
    1. **Predict** calories burned using Supervised Regression.
    2. **Segment** user behaviors using Unsupervised Clustering.
    """)

with col2:
    st.markdown("### 🌐 Domain")
    st.success("**Fitness Analytics / Health Tech / ML**")
    st.markdown("---")
    st.markdown("**Core Competencies:**")
    st.caption("Predictive Modeling • Behavioral Analytics • Data Lifecycle Management")

# Skills Section
st.markdown("### 🎓 Skills Takeaway")
skills = [
    "Data Preprocessing", "Feature Engineering", "Supervised Learning (Regression)",
    "Unsupervised Learning (Clustering)", "Dimensionality Reduction (PCA)",
    "Model Evaluation Metrics", "Business Interpretation of ML Results"
]
skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
st.markdown(skill_html, unsafe_allow_html=True)

st.markdown("---")

# The Two Streams (Supervised vs Unsupervised)
st.markdown("### 🧪 Machine Learning Approaches")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### 📈 Supervised Learning")
    st.write("**Goal:** Precise Calorie Prediction")
    st.write("- **Task:** Regression Modeling")
    st.write("- **Inputs:** Heart Rate, Duration, Experience, Hydration")
    st.write("- **Evaluation:** RMSE, MAE, R² Score")

with right_col:
    st.markdown("#### 🧩 Unsupervised Learning")
    st.write("**Goal:** Workout Pattern Discovery")
    st.write("- **Task:** User Segmentation & PCA")
    st.write("- **Process:** Dimensionality Reduction for Visualization")
    st.write("- **Evaluation:** Silhouette Score, Elbow Method")

st.markdown("---")

# Technical Stack Footer
st.markdown("### 🛠️ Technical Stack")
tech_stack = ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"]
tech_html = "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_stack])
st.markdown(tech_html, unsafe_allow_html=True)

# Sidebar for Navigation
st.sidebar.image("https://fitbit.com", width=150)
st.sidebar.title("Navigation")
st.sidebar.radio("Go to:", ["Home", "Data Explorer", "Regression Model", "Clustering Analysis"])

st.sidebar.markdown("---")
st.sidebar.write("Developed for **Health-Tech ML Portfolio**")
