import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from optigreen.features.pipeline import FeaturePipeline
from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster
from optigreen.forecasting.transformer_dataset import create_dataloaders
from optigreen.forecasting.transformer_model import TimeSeriesTransformer
from optigreen.evaluation.prob_metrics import evaluate_probabilistic_forecast
from optigreen.evaluation.forecast_metrics import evaluate_forecast
from optigreen.evaluation.inventory_sim import InventorySimulation
from optigreen.forecasting.xgboost_model import XGBoostForecaster

def run_phase3():
    print("=== OptiGreen-Chem Phase 3: Probabilistic Forecasting & Evaluation ===")
    data_dir = "data/synthetic"
    results_dir = "data/processed"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Data
    demand_df = pd.read_csv(f"{data_dir}/demand.csv")
    pipeline = FeaturePipeline()
    features_df = pipeline.build_features(demand_df)
    train_df, val_df, test_df = pipeline.time_based_split(features_df)
    
    # 2. Phase 2 XGBoost Baseline (Point Forecast for comparison)
    print("\n[1/4] Training Standard XGBoost (Phase 2 Baseline)...")
    xgb_base = XGBoostForecaster()
    xgb_base.train_with_validation(train_df, val_df, n_iter=1)
    base_preds = xgb_base.predict(test_df)
    
    # 3. Probabilistic XGBoost
    print("\n[2/4] Training Probabilistic XGBoost (P10, P50, P90)...")
    prob_xgb = ProbabilisticXGBoostForecaster()
    prob_xgb.train(train_df)
    pxgb_preds = prob_xgb.predict(test_df)
    
    # Evaluate P-XGB Point metrics using P50
    pxgb_preds['prediction'] = pxgb_preds['P50']
    pxgb_point_metrics = evaluate_forecast(pxgb_preds)
    pxgb_prob_metrics = evaluate_probabilistic_forecast(pxgb_preds)
    
    # 4. PyTorch Transformer
    print("\n[3/4] Training Temporal Transformer (P10, P50, P90)...")
    # Using small context for fast demonstration, usually 28
    ctx_len = 14
    hz_len = 7
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, context_length=ctx_len, horizon=hz_len, batch_size=256
    )
    
    # We have 10 features defined in SequentialDemandDataset
    model = TimeSeriesTransformer(num_features=10, d_model=32, nhead=2, num_layers=1, horizon=hz_len)
    trainer = pl.Trainer(max_epochs=2, accelerator='auto', enable_progress_bar=False, enable_checkpointing=False, logger=False)
    trainer.fit(model, train_loader, val_loader)
    
    print("      Transformer Inference...")
    model.eval()
    tft_preds_list = []
    # Simplified extraction of transformer predictions aligned with test dates
    # Real-world needs careful timestamp unrolling. For this phase, we compare XGBoost on the same dates.
    # We will just predict one step ahead for all test dates to match XGBoost's row-by-row shape.
    
    with torch.no_grad():
        for (region, product), group in test_df.groupby(['region_id', 'product_id']):
            group = group.reset_index(drop=True)
            for i in range(len(group)):
                # We need context. For simplicity in this demo simulation, we will use the test data's own past if i >= ctx_len,
                # else we pad with zeros or skip. 
                # Let's skip the first ctx_len rows per region/product in test set for clean eval.
                if i < ctx_len:
                    continue
                x = group.iloc[i-ctx_len:i][['day', 'day_of_week', 'week', 'month', 'is_weekend', 'rolling_mean_7', 'rolling_std_7', 'recent_trend', 'region_encoded', 'product_encoded']].values
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                out = model(x_t) # (1, horizon, 3)
                
                # We want 1-step ahead (horizon step 0)
                q_preds = out[0, 0, :].numpy()
                
                tft_preds_list.append({
                    'date': group.iloc[i]['date'],
                    'region_id': region,
                    'product_id': product,
                    'demand': group.iloc[i]['demand'],
                    'P10': max(0, q_preds[0]),
                    'P50': max(0, q_preds[1]),
                    'P90': max(0, q_preds[2])
                })
                
    tft_preds = pd.DataFrame(tft_preds_list)
    # Ensure P10 <= P50 <= P90
    tft_preds['P10'] = np.minimum(tft_preds['P10'], tft_preds['P50'])
    tft_preds['P90'] = np.maximum(tft_preds['P50'], tft_preds['P90'])
    tft_preds['prediction'] = tft_preds['P50']
    
    tft_point_metrics = evaluate_forecast(tft_preds)
    tft_prob_metrics = evaluate_probabilistic_forecast(tft_preds)
    
    # To keep comparison fair, we evaluate all models on the exact rows TFT predicted
    # Align all prediction sets on the same date/region/product keys
    eval_dates = tft_preds[['date', 'region_id', 'product_id']]
    
    # Merge XGB point predictions (keep lag_7 for naive baseline)
    base_preds_aligned = pd.merge(eval_dates, base_preds, on=['date', 'region_id', 'product_id'])
    # Preserve lag_7 from test_df for naive baseline
    lag_df = test_df[['date', 'region_id', 'product_id', 'lag_7']]
    base_preds_aligned = pd.merge(base_preds_aligned, lag_df, on=['date', 'region_id', 'product_id'])
    base_preds_aligned['lag_7'] = base_preds_aligned['lag_7'].fillna(0)
    
    pxgb_aligned = pd.merge(eval_dates, pxgb_preds, on=['date', 'region_id', 'product_id'])
    tft_aligned = tft_preds.copy()
    
    base_point_metrics = evaluate_forecast(base_preds_aligned)
    
    print("\n=== Forecasting Metrics Comparison ===")
    print(f"Standard XGBoost (Base) | WAPE: {base_point_metrics['global']['WAPE']:.4f}")
    print(f"Probabilistic XGBoost   | WAPE: {pxgb_point_metrics['global']['WAPE']:.4f} | PICP: {pxgb_prob_metrics['global']['PICP_80']:.3f} | MPIW: {pxgb_prob_metrics['global']['MPIW']:.1f}")
    print(f"Transformer (TFT-lite)  | WAPE: {tft_point_metrics['global']['WAPE']:.4f} | PICP: {tft_prob_metrics['global']['PICP_80']:.3f} | MPIW: {tft_prob_metrics['global']['MPIW']:.1f}")
    
    # Compute relative improvement/degradation vs XGBoost baseline
    xgb_wape = base_point_metrics['global']['WAPE']
    tft_wape = tft_point_metrics['global']['WAPE']
    rel_change = (xgb_wape - tft_wape) / xgb_wape * 100
    direction = 'improvement' if rel_change > 0 else 'regression'
    print(f"\nTFT vs XGBoost WAPE change: {rel_change:+.2f}% ({direction})")
    
    # 5. Inventory Simulation
    print("\n[4/4] Running Downstream Inventory Simulation...")
    # Build a unified simulation dataframe with all forecast columns
    sim_df = base_preds_aligned[['date', 'region_id', 'product_id', 'demand', 'lag_7']].copy()
    sim_df = pd.merge(sim_df, pxgb_aligned[['date', 'region_id', 'product_id', 'P50', 'P90']].rename(columns={'P50': 'xgb_p50', 'P90': 'xgb_p90'}), on=['date', 'region_id', 'product_id'])
    sim_df = pd.merge(sim_df, tft_aligned[['date', 'region_id', 'product_id', 'P90']].rename(columns={'P90': 'tft_p90'}), on=['date', 'region_id', 'product_id'])
    
    sim = InventorySimulation(sim_df)
    sim_res = sim.run_all_strategies({
        'Naive (lag_7)': 'lag_7',
        'XGB Point (P50)': 'xgb_p50',
        'XGB Probabilistic (P90)': 'xgb_p90',
        'TFT Probabilistic (P90)': 'tft_p90'
    })
    
    print("\n=== Inventory Simulation Results ===")
    print(sim_res[['Strategy Name', 'Service Level', 'Stockout Units', 'Holding Cost', 'Stockout Cost', 'Total Cost']].to_string(index=False))

    print("\nPhase 3 Execution Complete.")

if __name__ == "__main__":
    run_phase3()
