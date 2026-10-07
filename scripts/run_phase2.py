import os
import pandas as pd
from optigreen.features.pipeline import FeaturePipeline
from optigreen.forecasting.baselines import SeasonalNaiveBaseline, MovingAverageBaseline
from optigreen.forecasting.xgboost_model import XGBoostForecaster
from optigreen.evaluation.forecast_metrics import evaluate_forecast
from optigreen.evaluation.forecast_plots import plot_actual_vs_predicted, plot_error_distribution

def run_phase2():
    print("=== OptiGreen-Chem Phase 2: Forecasting Execution ===")
    
    data_dir = "data/synthetic"
    results_dir = "data/processed"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    demand_df = pd.read_csv(f"{data_dir}/demand.csv")
    print(f"Loaded {len(demand_df)} demand records.")
    
    # 2. Feature Engineering
    pipeline = FeaturePipeline()
    print("Building features...")
    features_df = pipeline.build_features(demand_df)
    
    # 3. Time-based Split
    train_df, val_df, test_df = pipeline.time_based_split(features_df)
    print(f"Chronological Split:")
    print(f"  Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    # We will evaluate all models on the TEST set
    eval_results = {}
    
    # 4. Seasonal Naive Baseline
    print("\nRunning Seasonal Naive Baseline (lag_7)...")
    naive = SeasonalNaiveBaseline(season_length=7)
    preds_naive = naive.predict(test_df)
    eval_results['Seasonal Naive'] = evaluate_forecast(preds_naive)
    
    # 5. Moving Average Baseline
    print("Running Moving Average Baseline (window=7)...")
    ma = MovingAverageBaseline(window=7)
    preds_ma = ma.predict(test_df)
    eval_results['Moving Average'] = evaluate_forecast(preds_ma)
    
    # 6. XGBoost
    print("Training XGBoost (with validation tuning)...")
    xgb_model = XGBoostForecaster()
    best_params = xgb_model.train_with_validation(train_df, val_df, n_iter=3)
    print(f"  Best params: {best_params}")
    preds_xgb = xgb_model.predict(test_df)
    eval_results['XGBoost'] = evaluate_forecast(preds_xgb)
    
    # 7. Reporting Metrics
    print("\n=== Evaluation Results (Global TEST Metrics) ===")
    for model_name, res in eval_results.items():
        m = res['global']
        print(f"{model_name:20s} | MAE: {m['MAE']:8.2f} | RMSE: {m['RMSE']:8.2f} | WAPE: {m['WAPE']:5.3f}")
        
    # 8. Visualizations
    print("\nGenerating visualizations...")
    plot_actual_vs_predicted(preds_xgb, "XGBoost Test Forecast", f"{results_dir}/xgb_actual_vs_pred.png")
    plot_error_distribution(preds_xgb, "XGBoost Test Forecast", f"{results_dir}/xgb_error_dist.png")
    print(f"Visualizations saved to {results_dir}/")
    
    print("\nPhase 2 Execution Complete.")

if __name__ == "__main__":
    run_phase2()
