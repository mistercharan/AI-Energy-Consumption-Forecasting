import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

FEATURE_COLS = [
    'hour','day','month','year','weekday','quarter',
    'day_of_year','week_of_year','is_weekend','is_peak_hour','season',
    'lag_1h','lag_2h','lag_3h','lag_6h','lag_12h',
    'lag_24h','lag_48h','lag_168h',
    'rolling_mean_3h','rolling_mean_6h','rolling_mean_24h','rolling_mean_7d',
    'rolling_std_24h','rolling_max_24h','rolling_min_24h'
]
TARGET_COL = 'energy_kwh'

def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    dates = df['datetime']
    print(f"[INFO] Dataset loaded. Shape: {df.shape}")
    return X, y, dates

def split_data(X, y, dates, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:]
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples:  {len(X_test)}")
    return X_train, X_test, y_train, y_test, dates_test

def train_model(X_train, y_train):
    print("[INFO] Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    model.fit(X_train, y_train)
    print("[INFO] Training complete!")
    return model

def evaluate_model(model, X_test, y_test, dates_test):
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    print("\n" + "="*45)
    print("      MODEL EVALUATION RESULTS")
    print("="*45)
    print(f"  RMSE  : {rmse:.4f} kWh")
    print(f"  MAE   : {mae:.4f} kWh")
    print(f"  R2    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    print("="*45)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.txt", "w") as f:
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*40 + "\n")
        f.write(f"RMSE  : {rmse:.4f} kWh\n")
        f.write(f"MAE   : {mae:.4f} kWh\n")
        f.write(f"R2    : {r2:.4f}\n")
        f.write(f"MAPE  : {mape:.2f}%\n")
    results_df = pd.DataFrame({
        'datetime': dates_test.values,
        'actual_kwh': y_test.values,
        'predicted_kwh': y_pred
    })
    results_df.to_csv("outputs/predictions.csv", index=False)
    print("[INFO] Predictions saved to outputs/predictions.csv")
    return y_pred, rmse, mae, r2

def save_model(model, path="models/xgb_energy_model.pkl"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to: {path}")

if __name__ == "__main__":
    X, y, dates = load_data("data/processed/featured_data.csv")
    X_train, X_test, y_train, y_test, dates_test = split_data(X, y, dates)
    model = train_model(X_train, y_train)
    y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test, dates_test)
    save_model(model)
    print("\n[SUCCESS] Training complete!")