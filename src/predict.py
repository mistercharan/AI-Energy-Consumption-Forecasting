import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

FEATURE_COLS = [
    'hour','day','month','year','weekday','quarter',
    'day_of_year','week_of_year','is_weekend','is_peak_hour','season',
    'lag_1h','lag_2h','lag_3h','lag_6h','lag_12h',
    'lag_24h','lag_48h','lag_168h',
    'rolling_mean_3h','rolling_mean_6h','rolling_mean_24h','rolling_mean_7d',
    'rolling_std_24h','rolling_max_24h','rolling_min_24h'
]

def load_model(path="models/xgb_energy_model.pkl"):
    model = joblib.load(path)
    print(f"[INFO] Model loaded from: {path}")
    return model

def forecast(model, data_path="data/processed/featured_data.csv", hours=48):
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    last_data = df[['datetime','energy_kwh']].tail(200).copy()
    history = list(last_data['energy_kwh'].values)
    last_time = pd.to_datetime(last_data['datetime'].iloc[-1])
    future_rows = []
    for i in range(1, hours + 1):
        ft = last_time + timedelta(hours=i)
        h = ft.hour
        row = {
            'hour': h, 'day': ft.day, 'month': ft.month,
            'year': ft.year, 'weekday': ft.weekday(),
            'quarter': (ft.month-1)//3+1,
            'day_of_year': ft.timetuple().tm_yday,
            'week_of_year': ft.isocalendar()[1],
            'is_weekend': int(ft.weekday() >= 5),
            'is_peak_hour': int((6<=h<=9) or (18<=h<=22)),
            'season': (0 if ft.month in [12,1,2] else
                       1 if ft.month in [3,4,5] else
                       2 if ft.month in [6,7,8] else 3),
            'lag_1h':   history[-1]   if len(history)>=1   else 0,
            'lag_2h':   history[-2]   if len(history)>=2   else 0,
            'lag_3h':   history[-3]   if len(history)>=3   else 0,
            'lag_6h':   history[-6]   if len(history)>=6   else 0,
            'lag_12h':  history[-12]  if len(history)>=12  else 0,
            'lag_24h':  history[-24]  if len(history)>=24  else 0,
            'lag_48h':  history[-48]  if len(history)>=48  else 0,
            'lag_168h': history[-168] if len(history)>=168 else 0,
            'rolling_mean_3h':  np.mean(history[-3:]),
            'rolling_mean_6h':  np.mean(history[-6:]),
            'rolling_mean_24h': np.mean(history[-24:]),
            'rolling_mean_7d':  np.mean(history[-168:]) if len(history)>=168 else np.mean(history),
            'rolling_std_24h':  np.std(history[-24:]),
            'rolling_max_24h':  np.max(history[-24:]),
            'rolling_min_24h':  np.min(history[-24:]),
        }
        future_rows.append(row)
        history.append(np.mean(history[-24:]))
    future_df = pd.DataFrame(future_rows)
    predictions = np.maximum(model.predict(future_df[FEATURE_COLS]), 0)
    forecast_df = pd.DataFrame({
        'datetime': [last_time + timedelta(hours=i) for i in range(1, hours+1)],
        'predicted_kwh': predictions
    })
    os.makedirs("outputs", exist_ok=True)
    forecast_df.to_csv("outputs/forecast_next48h.csv", index=False)
    print(f"[INFO] {hours}-hour forecast saved!")
    return forecast_df

if __name__ == "__main__":
    model = load_model()
    forecast_df = forecast(model, hours=48)
    print("\n[SUCCESS] Forecast complete!")
    print(forecast_df.head(10))