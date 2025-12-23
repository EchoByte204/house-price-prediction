import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create model directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

# Load data (RELATIVE PATH)
df = pd.read_csv("../data/mexico-real-estate-clean.csv")

# CHECK COLUMN NAMES
print(df.columns)

# CHANGE this to your actual price column
TARGET = "price_usd"   # <-- update if needed

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Save model and feature columns (RELATIVE PATHS)
joblib.dump(model, "../model/house_price_model.pkl")
joblib.dump(X.columns, "../model/feature_columns.pkl")

print("Model trained and saved successfully.")
