import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Page configuration
st.set_page_config(page_title="Fitbit Model Evaluation", layout="wide")
st.title("📊 Fitbit Activity Model Performance")

# --- UI Sidebar / Header ---
st.write("Click the button below to load data, clean it, and train multiple regression models.")

if st.button("🚀 Start Processing"):
    
    # 1. Load Data
    with st.spinner("Loading dataset..."):
        fitbitdf = pd.read_csv("Fitbit_dataset.csv")
        st.success("✅ Dataset loaded successfully!")

    # 2. Data Cleaning & Preprocessing
    with st.spinner("Cleaning and encoding data..."):
        fitbitdf.columns = fitbitdf.columns.str.lower().str.replace(' ', '_')
        fitbitdf.columns = fitbitdf.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        
        categorical_cols = ['gender', 'workout_type']
        df_encoded = pd.get_dummies(fitbitdf, columns=categorical_cols, drop_first=False)
        
        # Outlier Removal
        cols = ['weight_kg', 'height_m', 'fat_percentage']
        for col in cols:
            Q1 = df_encoded[col].quantile(0.25)
            Q3 = df_encoded[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_encoded = df_encoded[(df_encoded[col] >= lower) & (df_encoded[col] <= upper)]
        
        st.success(f"✅ Data cleaned. Remaining records: {df_encoded.shape[0]}")

    # 3. Feature Selection & Scaling
    with st.spinner("Preparing features..."):
        feature = df_encoded[['effective_met', 'base_met', 'session_duration_hours','weight_kg','bmi','height_m']]
        target = df_encoded['calories_burned_kcal']
        
        x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=4)
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        st.success("✅ Features scaled and split into Train/Test sets.")

    # 4. Model Training
    with st.spinner("Training models (Linear, KNN, RF, DT, XGB, SVR)..."):
        # Linear Regression
        linear_model = LinearRegression().fit(x_train_scaled, y_train)
        y_pred_standard = linear_model.predict(x_test_scaled)

        # KNN
        knn_model = KNeighborsRegressor(n_neighbors=5).fit(x_train_scaled, y_train)
        y_pred_knn = knn_model.predict(x_test_scaled)

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(x_train_scaled, y_train)
        y_pred_rf = rf_model.predict(x_test_scaled)

        # Decision Tree
        dt_model = DecisionTreeRegressor(random_state=42).fit(x_train_scaled, y_train)
        y_pred_dt = dt_model.predict(x_test_scaled)

        # XGBoost
        xgb_model = XGBRegressor(random_state=42).fit(x_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(x_test_scaled)

        # SVR
        svr_model = SVR(kernel='rbf').fit(x_train_scaled, y_train)
        y_pred_svr = svr_model.predict(x_test_scaled)
        
        st.success("✅ All models trained successfully!")

    # 5. Results Compilation
    def get_results_df(y_test, predictions):
        results = {'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R-squared': []}
        for model_name, y_pred in predictions.items():
            results['Model'].append(model_name)
            results['MAE'].append(metrics.mean_absolute_error(y_test, y_pred))
            results['MSE'].append(metrics.mean_squared_error(y_test, y_pred))
            results['RMSE'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
            results['R-squared'].append(r2_score(y_test, y_pred))
        return pd.DataFrame(results).sort_values(by='R-squared', ascending=False)

    predictions_map = {
        'Linear Regression': y_pred_standard,
        'KNN Regressor': y_pred_knn,
        'Random Forest Regressor': y_pred_rf,
        'Decision Tree Regressor': y_pred_dt,
        'XGBoost Regressor': y_pred_xgb,
        'Support Vector Regressor': y_pred_svr
    }

    df_results = get_results_df(y_test, predictions_map)

    # 6. Displaying Results
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Results Table")
        st.dataframe(df_results.style.highlight_max(axis=0, subset=['R-squared'], color='#2e7d32'), use_container_width=True)

    with col2:
        st.subheader("R-squared Comparison")
        st.bar_chart(df_results.set_index('Model')['R-squared'])

    best_model = df_results.iloc[0]['Model']
    st.balloons()
    st.success(f"🏆 **{best_model}** is the best performing model based on R-squared.")
