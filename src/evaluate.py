import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("../data/mexico-real-estate-clean.csv")

# CHANGE this if your column name is different
TARGET = "price_usd"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split (same random_state as training!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("../model/house_price_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Evaluation Results")
print("------------------")
print(f"MAE  : {mae}")
print(f"RMSE : {rmse}")
print(f"R²   : {r2}")
