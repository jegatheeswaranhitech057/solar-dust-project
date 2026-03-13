"""
train_model.py
--------------

Train ML model to predict Expected Energy Generation
using solar plant operational data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# ------------------------------------------------
# Paths
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "solar data.csv")

MODEL_PATH = os.path.join(BASE_DIR, "energy_model.pkl")


# ------------------------------------------------
# Feature Columns
# ------------------------------------------------

FEATURE_COLS = [
    "ambient_temp",
    "module_temp",
    "tilt_radiation",
    "peak_tilt_irradiation",
    "wind_speed",
    "plant_peak_power"
]

TARGET_COL = "energy_generation"


# ------------------------------------------------
# Load Dataset
# ------------------------------------------------

def load_data(path):

    df = pd.read_csv(path)

    # Rename dataset columns
    df = df.rename(columns={
        "AMBIENT TEMP (*C)": "ambient_temp",
        "MODULE TEMP (*C)": "module_temp",
        "TILT RADIATION (Wh/m2)": "tilt_radiation",
        "PEAK TILT IRRADIATION (Wh/m2)": "peak_tilt_irradiation",
        "WIND SPEED (Km/Hr)": "wind_speed",
        "PLANT PEAK POWER (KW)": "plant_peak_power",
        "ENERGY GENERATION (KWH)": "energy_generation"
    })

    # Select only required columns
    df = df[FEATURE_COLS + [TARGET_COL]]

    # Convert everything to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove rows with NaN values
    df = df.dropna()

    print("Rows after cleaning:", len(df))

    return df

# ------------------------------------------------
# Train Model
# ------------------------------------------------

def train(df):

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel trained successfully")
    print("R2 Score :", round(r2,3))
    print("MAE :", round(mae,3))

    return model


# ------------------------------------------------
# Save Model
# ------------------------------------------------

def save_model(model, path):

    joblib.dump(model, path)

    print("\nModel saved at:", path)


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print("Rows loaded:", len(df))

    print("Training model...")

    model = train(df)

    save_model(model, MODEL_PATH)