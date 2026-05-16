import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Workout Intensity Discovery", layout="wide")
st.title("🏃‍♂️ Hidden Workout Pattern Identification")
st.markdown("Discovering intensity levels through physiological data clustering.")

# Load Data
df = pd.read_csv('cleaned.csv')

# Drop target variable (Workout_Type) from df
df = df.drop(columns=['workout_type'])

features = ['max_bpm', 'avg_bpm', 'calories_burned_kcal']
X = df[features].dropna()

# 2. Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Sidebar Controls to discover the no.of patterns
num_clusters = st.sidebar.slider("Number of Patterns to Find", 2, 5, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df.loc[X.index, 'Pattern_Group'] = labels

# Calculate metrics once
score = silhouette_score(X_scaled, labels)
#st.sidebar.metric("Silhouette Score", f"{score:.3f}")

# --- NEW: TABBED NAVIGATION ---
tab1, tab2 = st.tabs(["📊 Discovery Dashboard", "🧪 Model Validation"])

with tab1:
    st.subheader(f"Identified {num_clusters} Hidden Workout Patterns")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(df, x='avg_bpm', y='calories_burned_kcal', 
                          color='Pattern_Group', size='max_bpm', 
                          title="Workout Intensity vs. Average Heart Rate")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:# shows the datatable
        summary = df.groupby('Pattern_Group')[features].mean()
        st.write("Pattern Characteristics (Averages):")
        st.dataframe(summary.style.background_gradient(cmap='Blues').format("{:.2f}"))

    st.info("""
    **Interpreting the Patterns:**
    - 0 (Dark Blue) - High Burn Rate / High BPM: Likely HIIT or heavy cardio - Running, HIIT, Cycling.
    - 1 (Mid Blue) - Moderate Burn Rate / Moderate BPM: Likely Strength Training or steady-state endurance - Weightlifting, Sprints.
    - 2 (Lightest) - Low Burn Rate / Low BPM: Likely Yoga , flexibility, walking, or recovery Light Cardio.
    """)

with tab2:
    st.subheader("Silhouette Analysis & Performance")
    
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.metric("Overall Silhouette Score", f"{score:.3f}")
        if score > 0.5:
            st.success("✅ **Strong Clustering**")
        elif score > 0.2:
            st.warning("⚠️ **Moderate Clustering**")
        else:
            st.error("❗ **Weak Clustering**")
    
    with col_b:
        st.markdown("""
        **What does this mean?**
        The silhouette score - A metric used in machine learning to evaluate the quality of a clustering algorithm(like K Means). 
        Higher is better!
        - **Near 1:** Clusters are very dense and well-separated.
        - **Near 0:** Clusters are overlapping.
        - **Negative:** Workouts might be assigned to the wrong group.
          """)
        
    
       

    # Interactive Silhouette Chart
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    fig_sil = go.Figure()
    y_lower = 10
    

    for i in range(num_clusters):
        ith_values = sample_silhouette_values[labels == i]
        ith_values.sort()
        size_cluster_i = ith_values.shape[0]
        y_upper = y_lower + size_cluster_i

        fig_sil.add_trace(go.Scatter(
            x=ith_values, y=np.arange(y_lower, y_upper),
            fill='tozerox', name=f'Pattern {i}',
            mode='lines', line=dict(width=0.5),
            text=[f"Cluster {i}"] * size_cluster_i,
            hoverinfo="x+text"
        ))
        y_lower = y_upper + 10 

    fig_sil.add_vline(x=score, line_dash="dash", line_color="red", 
                      annotation_text=f"Avg: {score:.2f}", annotation_position="top right")

    fig_sil.update_layout(
        title="Silhouette Coefficients per Cluster",
        xaxis_title="Silhouette Coefficient Value",
        yaxis_showticklabels=False,
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_sil, use_container_width=True)
    
    
    
    
