import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_and_clean, save_processed
from features   import engineer_features, save_featured
from train      import load_data, split_data, train_model, evaluate_model, save_model
from predict    import load_model, forecast
from visualize  import generate_all_plots

def run_pipeline():
    print("\n" + "="*55)
    print("   AI-POWERED ENERGY CONSUMPTION FORECASTING")
    print("="*55 + "\n")
    start = time.time()

    print("PHASE 1: Preprocessing...")
    df_clean = load_and_clean("data/raw/energy_data.csv")
    save_processed(df_clean, "data/processed/cleaned_hourly.csv")

    print("\nPHASE 2: Feature Engineering...")
    df_featured = engineer_features(df_clean)
    save_featured(df_featured, "data/processed/featured_data.csv")

    print("\nPHASE 3: Model Training...")
    X, y, dates = load_data("data/processed/featured_data.csv")
    X_train, X_test, y_train, y_test, dates_test = split_data(X, y, dates)
    model = train_model(X_train, y_train)
    y_pred, rmse, mae, r2 = evaluate_model(model, X_test, y_test, dates_test)
    save_model(model)

    print("\nPHASE 4: Forecasting...")
    trained_model = load_model()
    forecast_df = forecast(trained_model, hours=48)

    print("\nPHASE 5: Visualizations...")
    generate_all_plots()

    elapsed = time.time() - start
    print("\n" + "="*55)
    print("          PIPELINE COMPLETE!")
    print("="*55)
    print(f"  Time   : {elapsed:.1f} seconds")
    print(f"  R2     : {r2:.4f}")
    print(f"  Graphs : outputs/graphs/")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_pipeline()