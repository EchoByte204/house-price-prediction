# House Price Prediction

## Overview
This project implements an end-to-end machine learning pipeline to predict house prices based on property characteristics such as size, geographic location, and property type. The primary objective of this project is to demonstrate a clean, reproducible, and well-structured machine learning workflow rather than to achieve unrealistic prediction accuracy.

---

## Problem Statement
Given historical real estate data, the task is to predict the selling price of a house using supervised machine learning techniques. This is a regression problem where the target variable is the house price.

---

## Dataset
- Real estate dataset from Mexico
- Data has been cleaned and preprocessed prior to modeling
- Features include:
  - Property area (square meters)
  - Latitude and longitude
  - State
  - Property type (house, apartment, etc.)

The cleaned dataset is located at: data/mexico-real-estate-clean.csv


---

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to understand the dataset and guide model selection. The EDA includes:
- Feature distribution analysis
- Outlier detection
- Correlation analysis
- Identification of important predictors

The EDA notebook can be found here: notebooks/eda.ipynb


---

## Model Development
Multiple machine learning models were evaluated:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Final Model Selection
**Random Forest Regressor** was selected as the final model due to:
- Strong performance on unseen test data
- Ability to capture non-linear relationships
- Robustness to noise and mixed feature types

---

## Model Performance
Model evaluation was performed using an 80/20 train-test split.

**Evaluation Metrics:**
- R² Score: ~0.54
- Mean Absolute Error (MAE): ~31,000
- Root Mean Squared Error (RMSE): ~45,000

These metrics are realistic for real estate price prediction, where many influencing factors are not available in the dataset.

---

## Feature Importance
Feature importance analysis using Random Forest indicates that:
- Property area (m²) is the most influential feature
- Geographic location (latitude and longitude) significantly affects price
- State and property type contribute moderately

This confirms that the model learns meaningful real-world patterns.

---

## Project Structure
house-price-prediction/
│
├── data/
│ └── mexico-real-estate-clean.csv
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ ├── feature_importance.py
│ └── cli_predict.py
│
├── model/ # generated locally (not committed)
├── README.md


---

## How to Run

### Train/Evaluate/predict the Model
```bash
python src/train.py
python src/evaluate.py
python src/cli_predict.py


