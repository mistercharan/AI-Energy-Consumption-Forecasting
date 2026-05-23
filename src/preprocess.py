import pandas as pd
import numpy as np
import os

def load_and_clean(filepath):
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(
        filepath,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        infer_datetime_format=True,
        na_values=['?'],
        low_memory=False
    )
    print(f"[INFO] Raw shape: {df.shape}")
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    df['energy_kwh'] = df['Global_active_power'].astype(float) * (1/60)
    df_hourly = df['energy_kwh'].resample('H').sum()
    df_hourly = df_hourly.reset_index()
    df_hourly.columns = ['datetime', 'energy_kwh']
    df_hourly = df_hourly[df_hourly['energy_kwh'] > 0]
    print(f"[INFO] Cleaned hourly shape: {df_hourly.shape}")
    print(f"[INFO] Date range: {df_hourly['datetime'].min()} to {df_hourly['datetime'].max()}")
    return df_hourly

def save_processed(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved to: {output_path}")

if __name__ == "__main__":
    df = load_and_clean("data/raw/energy_data.csv")
    save_processed(df, "data/processed/cleaned_hourly.csv")
    print("\n[SUCCESS] Preprocessing complete!")
    print(df.head(10))