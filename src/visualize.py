import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import os

plt.rcParams['figure.dpi'] = 150
SAVE_DIR = "outputs/graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURE_COLS = [
    'hour','day','month','year','weekday','quarter',
    'day_of_year','week_of_year','is_weekend','is_peak_hour','season',
    'lag_1h','lag_2h','lag_3h','lag_6h','lag_12h',
    'lag_24h','lag_48h','lag_168h',
    'rolling_mean_3h','rolling_mean_6h','rolling_mean_24h','rolling_mean_7d',
    'rolling_std_24h','rolling_max_24h','rolling_min_24h'
]

def plot_actual_vs_predicted():
    df = pd.read_csv("outputs/predictions.csv", parse_dates=['datetime'])
    df_plot = df.tail(168)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(df_plot['datetime'], df_plot['actual_kwh'], label='Actual', color='#2196F3', linewidth=1.5)
    axes[0].plot(df_plot['datetime'], df_plot['predicted_kwh'], label='Predicted', color='#FF5722', linewidth=1.5, linestyle='--')
    axes[0].set_title('Energy Consumption: Actual vs Predicted (Last 7 Days)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Energy (kWh)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(df['actual_kwh'], df['predicted_kwh'], alpha=0.3, color='#4CAF50', s=10)
    max_val = max(df['actual_kwh'].max(), df['predicted_kwh'].max())
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Energy (kWh)')
    axes[1].set_ylabel('Predicted Energy (kWh)')
    axes[1].set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/actual_vs_predicted.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] actual_vs_predicted.png")

def plot_energy_heatmap():
    df = pd.read_csv("data/processed/featured_data.csv", parse_dates=['datetime'])
    pivot = df.pivot_table(values='energy_kwh', index='hour', columns='weekday', aggfunc='mean')
    pivot.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Energy Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/energy_heatmap.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] energy_heatmap.png")

def plot_forecast():
    forecast_df = pd.read_csv("outputs/forecast_next48h.csv", parse_dates=['datetime'])
    actual_df   = pd.read_csv("outputs/predictions.csv", parse_dates=['datetime'])
    recent = actual_df.tail(72)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent['datetime'], recent['actual_kwh'], label='Historical', color='#2196F3', linewidth=2)
    ax.plot(forecast_df['datetime'], forecast_df['predicted_kwh'], label='Forecast (48h)', color='#FF9800', linewidth=2.5, linestyle='--', marker='o', markersize=3)
    ax.axvspan(forecast_df['datetime'].iloc[0], forecast_df['datetime'].iloc[-1], alpha=0.08, color='orange')
    ax.set_title('Energy Consumption Forecast — Next 48 Hours', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date & Time')
    ax.set_ylabel('Energy (kWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/forecast_next48h.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] forecast_next48h.png")

def plot_feature_importance():
    model = joblib.load("models/xgb_energy_model.pkl")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importance})
    feat_df = feat_df.sort_values('importance', ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feat_df['feature'], feat_df['importance'], color='#2196F3', edgecolor='white')
    ax.set_title('Top 15 Feature Importances (XGBoost)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/feature_importance.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] feature_importance.png")

def plot_monthly_trend():
    df = pd.read_csv("data/processed/featured_data.csv", parse_dates=['datetime'])
    monthly = df.groupby(['year','month'])['energy_kwh'].mean().reset_index()
    monthly['period'] = pd.to_datetime(monthly[['year','month']].assign(day=1))
    monthly.sort_values('period', inplace=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly['period'], monthly['energy_kwh'], color='#4CAF50', alpha=0.7, width=20)
    ax.plot(monthly['period'], monthly['energy_kwh'].rolling(3, center=True).mean(), color='#E91E63', linewidth=2.5, label='3-Month Trend')
    ax.set_title('Monthly Average Energy Consumption', fontsize=13, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Avg Energy (kWh/hour)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/monthly_trend.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] monthly_trend.png")

def plot_error_distribution():
    df = pd.read_csv("outputs/predictions.csv")
    df['error'] = df['actual_kwh'] - df['predicted_kwh']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['error'], bins=80, color='#9C27B0', alpha=0.7, edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Error Distribution', fontweight='bold')
    axes[0].set_xlabel('Error (kWh)')
    axes[0].grid(True, alpha=0.3)
    pct_error = ((df['actual_kwh'] - df['predicted_kwh']) / (df['actual_kwh'] + 1e-8)) * 100
    axes[1].hist(pct_error, bins=80, color='#FF9800', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Percentage Error Distribution', fontweight='bold')
    axes[1].set_xlabel('% Error')
    axes[1].set_xlim(-50, 50)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('Model Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/error_distribution.png", bbox_inches='tight')
    plt.close()
    print("[SAVED] error_distribution.png")

def generate_all_plots():
    print("\n[INFO] Generating all visualizations...")
    plot_actual_vs_predicted()
    plot_energy_heatmap()
    plot_forecast()
    plot_feature_importance()
    plot_monthly_trend()
    plot_error_distribution()
    print(f"\n[SUCCESS] All 6 plots saved to: {SAVE_DIR}/")

if __name__ == "__main__":
    generate_all_plots()