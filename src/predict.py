import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("../model/house_price_model.pkl")
feature_columns = joblib.load("../model/feature_columns.pkl")

# Example new house input (CHANGE VALUES)
new_house = {
    "area_m2": 120,
    "lat": 19.43,
    "lon": -99.13,
    "property_type_house": 1,
    "state_Distrito Federal": 1
}

# Convert to DataFrame
input_df = pd.DataFrame([new_house])

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training data
input_df = input_df[feature_columns]

# Predict
predicted_price = model.predict(input_df)[0]

print("Predicted House Price:", predicted_price)
