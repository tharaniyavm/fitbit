import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Workout Intensity Discovery", layout="wide")
st.title("🏃‍♂️ Hidden Workout Pattern Identification")
st.markdown("Discovering intensity levels through physiological data clustering.")

df = pd.read_csv('cleaned.csv')


# Selecting physiological & intensity features for clustering
features = ['max_bpm', 'avg_bpm', 'calories_burned_kcal']
X = df[features].dropna()

# 4. Data Scaling (Critical for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Unsupervised Clustering
num_clusters = st.sidebar.slider("Number of Patterns to Find", 2, 5, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df.loc[X.index, 'Pattern_Group'] = kmeans.fit_predict(X_scaled)

# 6. Visualizing the Patterns
st.subheader(f"Identified {num_clusters} Hidden Workout Patterns")

col1, col2 = st.columns(2)

with col1:
    # Intensity vs Heart Rate Plot
    fig1 = px.scatter(df, x='avg_bpm', y='calories_burned_kcal', color='Pattern_Group',
                        size='max_bpm', title="Workout Intensity vs. Average Heart Rate")
    st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    # Pattern Summary
    summary = df.groupby('Pattern_Group')[features].mean()
    #summary.style.format("{:.2f}")
    st.write("Pattern Characteristics (Averages):")
    st.dataframe(summary.style.background_gradient(cmap='Blues').format("{:.2f}"))

# 7. Insights Section
st.info("""
**Interpreting the Patterns:**
- **High Burn Rate / High BPM**: Likely HIIT or heavy cardio.
- **Low Burn Rate / Low BPM**: Likely flexibility, walking, or recovery.
- **Moderate Burn Rate / Moderate BPM**: Likely steady-state endurance.
""")
