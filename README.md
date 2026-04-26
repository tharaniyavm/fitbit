# Fitbit Fitness Analytics

A Streamlit-based fitness analytics project for calorie burn prediction and workout pattern analysis using Fitbit data.

## Overview

This project demonstrates a health-tech machine learning application that:
- predicts calories burned using a regression model,
- explores model performance,
- supports a Streamlit interface with Multi-page navigation,
- uses Fitbit workout and body data for analysis.

## Key Files

- `Home.py` - Main Streamlit landing page with project summary and navigation.
- `pages/2_CaloriePredictor.py` - Calorie prediction interface using a trained model.
- `pages/1_ModelPerformanceMetrics.py` - Model training and evaluation pipeline.
- `Fitbit_dataset.csv` - Primary Fitbit dataset used for preprocessing and modeling.
- `linear_regression_model.pkl` - Saved regression model for inference.
- `scaler.pkl` - Saved feature scaler used during training.
- `calorie_prediction.ipynb` - Exploratory notebook for data analysis and modeling.
- `cleaned.csv` - Processed dataset version.

## Requirements

Install dependencies using pip:

```bash
pip install streamlit pandas numpy scikit-learn xgboost
```

## Run the App

From the project root directory:

```bash
streamlit run Home.py
```

Then open the local URL shown in the terminal.

## Notes

- Ensure `Fitbit_dataset.csv`, `linear_regression_model.pkl`, and `scaler.pkl` are available in the project root before using the predictor page.
- The app is designed for local experimentation and demonstration of fitness analytics workflows.
