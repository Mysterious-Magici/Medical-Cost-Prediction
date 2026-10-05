# Medical Cost Prediction Pipeline

An end-to-end machine learning project predicting annual health insurance charges based on patient demographics and health attributes. This repository demonstrates data cleaning, exploratory data analysis (EDA), preprocessing, and the implementation of regression models.

## Project Overview
Linear Regression provides a baseline, but the core analysis utilizes **XGBoost** to capture critical non-linear interactions in healthcare data—specifically the exponential cost multiplication when high BMI intersects with active smoking.

## Repository Contents
* **`Insurance Cost Prediction.ipynb`**: Complete Jupyter Notebook containing EDA, preprocessing, cross-validation, and model evaluation.
* **`insurance.csv`**: The raw medical cost dataset.
* **`insurance_charge_predictor.joblib`**: Serialized baseline Linear Regression model.
* **`tuned_xgboost_insurance_model.joblib`**: Serialized optimized XGBoost model.

## Tech Stack
* **Language:** Python
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Serialization:** Joblib

## Key Findings
The tuned XGBoost model significantly outperformed the linear baseline by accurately mapping the interaction between obesity (BMI > 30) and smoking. The algorithm successfully isolates this high-risk subgroup, reducing the Mean Absolute Error (MAE) and explaining approximately 87-89% of the variance ($R^2$ Score) in medical charges.

## How to Run
1. Clone the repository.
2. Install requirements: `pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib`
3. Load the pre-trained model for inference:
```python
import joblib
import pandas as pd

model_bundle = joblib.load('tuned_xgboost_insurance_model.joblib')
model = model_bundle['model']
# Pass formatted Pandas DataFrame to model.predict(df)