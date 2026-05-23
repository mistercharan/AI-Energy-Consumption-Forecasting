import pandas as pd
import numpy as np
import os

def add_datetime_features(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['quarter'] = df['datetime'].dt.quarter
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].apply(
        lambda h: 1 if (6 <= h <= 9) or (18 <= h <= 22) else 0
    )
    df['season'] = df['month'].apply(lambda m:
        0 if m in [12,1,2] else
        1 if m in [3,4,5] else
        2 if m in [6,7,8] else 3
    )
    print("[INFO] Datetime features added.")
    return df

def add_lag_features(df):
    df = df.copy()
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df['lag_1h'] = df['energy_kwh'].shift(1)
    df['lag_2h'] = df['energy_kwh'].shift(2)
    df['lag_3h'] = df['energy_kwh'].shift(3)
    df['lag_6h'] = df['energy_kwh'].shift(6)
    df['lag_12h'] = df['energy_kwh'].shift(12)
    df['lag_24h'] = df['energy_kwh'].shift(24)
    df['lag_48h'] = df['energy_kwh'].shift(48)
    df['lag_168h'] = df['energy_kwh'].shift(168)
    df['rolling_mean_3h'] = df['energy_kwh'].shift(1).rolling(window=3).mean()
    df['rolling_mean_6h'] = df['energy_kwh'].shift(1).rolling(window=6).mean()
    df['rolling_mean_24h'] = df['energy_kwh'].shift(1).rolling(window=24).mean()
    df['rolling_mean_7d'] = df['energy_kwh'].shift(1).rolling(window=168).mean()
    df['rolling_std_24h'] = df['energy_kwh'].shift(1).rolling(window=24).std()
    df['rolling_max_24h'] = df['energy_kwh'].shift(1).rolling(window=24).max()
    df['rolling_min_24h'] = df['energy_kwh'].shift(1).rolling(window=24).min()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    print(f"[INFO] Lag features added. Shape: {df.shape}")
    return df

def engineer_features(df):
    df = add_datetime_features(df)
    df = add_lag_features(df)
    return df

def save_featured(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Featured data saved to: {output_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_hourly.csv")
    df_featured = engineer_features(df)
    save_featured(df_featured, "data/processed/featured_data.csv")
    print("\n[SUCCESS] Feature engineering complete!")
    print(f"Total features: {len(df_featured.columns)}")