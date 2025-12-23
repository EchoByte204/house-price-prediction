import joblib
import pandas as pd

# Load model and feature columns
model = joblib.load("../model/house_price_model.pkl")
feature_columns = joblib.load("../model/feature_columns.pkl")

print("House Price Prediction CLI")
print("---------------------------")

# User input
area = float(input("Enter area in square meters: "))
lat = float(input("Enter latitude: "))
lon = float(input("Enter longitude: "))

property_type = input("Property type (house/apartment): ").strip().lower()
state = input("State (exact name as in dataset, e.g. Distrito Federal): ").strip()

# Base input dictionary
input_data = {
    "area_m2": area,
    "lat": lat,
    "lon": lon
}

# Handle property type encoding
property_col = f"property_type_{property_type}"
if property_col in feature_columns:
    input_data[property_col] = 1

# Handle state encoding
state_col = f"state_{state}"
if state_col in feature_columns:
    input_data[state_col] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# Predict
predicted_price = model.predict(input_df)[0]

print("\nPredicted House Price:", round(predicted_price, 2))
