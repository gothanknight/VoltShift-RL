"""
Supervised Machine Learning Forecaster for VoltShift-RL.
Trains a Histogram-based Gradient Boosting Regressor on lagged historical
irradiance to forecast solar generation 1 hour (4 timesteps) ahead.
"""

import polars as pl
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from src.config import TELEMETRY_PARQUET_PATH, FORECASTER_MODEL_PATH

FEATURE_COLS = [
    "hour",
    "ambient_temp_c",
    "solar_current",
    "solar_lag_15m",
    "solar_lag_30m",
    "solar_lag_60m"
]

def prepare_forecasting_data(parquet_path=TELEMETRY_PARQUET_PATH):
    """
    Loads parquet telemetry and constructs supervised lag features
    to predict solar irradiance 1 hour (4 timesteps) into the future.
    """
    df = pl.read_parquet(parquet_path)

    # Construct temporal lag features using Polars shifts
    df = df.with_columns([
        pl.col("solar_kw").alias("solar_current"),
        pl.col("solar_kw").shift(1).fill_null(0.0).alias("solar_lag_15m"),
        pl.col("solar_kw").shift(2).fill_null(0.0).alias("solar_lag_30m"),
        pl.col("solar_kw").shift(4).fill_null(0.0).alias("solar_lag_60m"),
        pl.col("solar_kw").shift(-4).fill_null(0.0).alias("target_solar_next_1h")  # 1-hour lookahead
    ])

    # Convert to Pandas / NumPy for scikit-learn
    pdf = df.to_pandas()
    
    # 80/20 Chronological Train/Test split (first 24 days train, last 6 days test)
    split_idx = int(len(pdf) * 0.8)
    
    X_train = pdf.iloc[:split_idx][FEATURE_COLS]
    y_train = pdf.iloc[:split_idx]["target_solar_next_1h"]
    
    X_test = pdf.iloc[split_idx:][FEATURE_COLS]
    y_test = pdf.iloc[split_idx:]["target_solar_next_1h"]
    
    return X_train, y_train, X_test, y_test

def train_forecaster():
    """
    Trains the Gradient Boosting Regressor on CPU and serializes to disk.
    """
    print("\n--- Training Supervised ML Forecaster ---")
    X_train, y_train, X_test, y_test = prepare_forecasting_data()
    
    # Fast, CPU-friendly histogram gradient boosting
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=5,
        learning_rate=0.08,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation on held-out 6-day test window
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Test Set Window:                Last 6 days (576 timesteps)")
    print(f"Test MAE (Mean Absolute Error): {mae:.3f} kW")
    print(f"Test R² Score:                  {r2:.3f}")
    
    # Save the trained model artifact
    joblib.dump(model, FORECASTER_MODEL_PATH)
    print(f"Forecaster saved successfully to: {FORECASTER_MODEL_PATH}\n")

if __name__ == "__main__":
    train_forecaster()