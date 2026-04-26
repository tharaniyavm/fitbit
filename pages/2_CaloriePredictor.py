import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Calorie Predictor", page_icon="⚡")

def prediction_page():
    st.title("🏃 Calorie Burn Predictor")
    st.markdown("Enter your workout details below to estimate calories burned using the trained model.")

    # --- LOAD PRE-TRAINED MODEL & SCALER ---
    try:
        with open('linear_regression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Note: You usually need the scaler used during training to get accurate results
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    except FileNotFoundError:
        st.error("Model or Scaler file not found. Please ensure 'linear_regression_model.pkl' and 'scaler.pkl' exist.")
        return

    # --- USER INPUT SECTION ---
    st.markdown("### 📊 Workout & Body Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        effective_met = st.number_input("Effective MET (Intensity)", min_value=1.0, max_value=30.0, value=7.5, step=0.1)
        base_met = st.number_input("Base MET", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
        session_duration_hours = st.number_input("Session Duration (Hours)", min_value=0.01, max_value=24.0, value=1.0, step=0.25)
        
    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=70.0, step=0.5)
        height_m = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
        bmi = st.number_input("BMI", min_value=5.0, max_value=60.0, value=22.0, step=0.1)

    st.markdown("---")

    # --- INFERENCE ---
    if st.button("🚀 Calculate Calories Burned", use_container_width=True):
        # Create input array (Ensure order matches the training features)
        input_data = np.array([[effective_met, base_met, session_duration_hours, weight_kg, bmi, height_m]])
        
        # Apply the loaded scaler
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        # Display Result
        st.success(f"### Estimated Burn: {prediction[0]:.2f} kcal")
        st.metric(label="Total Energy Expenditure", value=f"{prediction[0]:.1f} kcal")

if __name__ == "__main__":
    prediction_page()
