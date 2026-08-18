"""
Phase 5 Execution Script: GNN/GAT Supply Chain Risk Intelligence

1. Loads Phase 1/2 synthetic data and Phase 3 forecasts
2. Builds the temporal graph dataset
3. Trains Logistic Regression and XGBoost baselines
4. Trains GCN and GAT
5. Runs disruption propagation analysis
6. Saves models, metrics, and visualizations
"""
import os
import yaml
import torch
import pandas as pd
import numpy as np

from optigreen.graph.graph_dataset import build_graph_dataset
from optigreen.graph.baselines import BaselineModels
from optigreen.graph.gcn_model import GCNRiskModel
from optigreen.graph.gat_model import GATRiskModel
from optigreen.graph.trainer import GNNTrainer
from optigreen.graph.disruption_analysis import DisruptionAnalyzer
from optigreen.graph.gnn_plots import (
    plot_metrics_comparison, plot_disruption_stress_test,
    plot_downstream_propagation, plot_graph_topology_risk
)

# Directories
DATA_DIR = "data/synthetic"
PROC_DIR = "data/processed"
OUT_DIR = "data/phase5"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=== OptiGreen-Chem Phase 5: GNN Risk Intelligence ===")
    
    # 1. Load config
    with open("configs/gnn_config.yaml", "r") as f:
        config = yaml.safe_load(f)['gnn']
        
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 2. Load data
    print("\nLoading synthetic data and forecasts...")
    demand_df = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"))
    plants_df = pd.read_csv(os.path.join(DATA_DIR, "plants.csv"))
    wh_df = pd.read_csv(os.path.join(DATA_DIR, "warehouses.csv"))
    regions_df = pd.read_csv(os.path.join(DATA_DIR, "regions.csv"))
    routes_df = pd.read_csv(os.path.join(DATA_DIR, "routes.csv"))
    
    # 3. Build graph dataset
    print("\nBuilding Temporal Graph Dataset...")
    train_data, val_data, test_data, metadata = build_graph_dataset(
        demand_df, plants_df, wh_df, regions_df, routes_df, pd.DataFrame(),
        window_weeks=config['window_weeks'],
        train_frac=config['train_frac'],
        val_frac=config['val_frac'],
        seed=config['seed']
    )
    
    # Update config with actual dimensions
    config['node_feature_dim'] = metadata['node_feature_dim']
    config['edge_feature_dim'] = metadata['edge_feature_dim']
    
    # 4. Train Baselines
    print("\n--- Training Classical Baselines ---")
    baselines = BaselineModels(pos_weight=config['pos_weight'])
    baselines.train(train_data, val_data)
    baseline_metrics = baselines.evaluate(test_data)
    print("Baseline Test PR-AUC:")
    print(f"  Logistic: {baseline_metrics['Logistic']['PR_AUC']:.3f}")
    print(f"  XGBoost:  {baseline_metrics['XGBoost']['PR_AUC']:.3f}")

    # 5. Train GCN
    print("\n--- Training GCN ---")
    gcn = GCNRiskModel(
        node_feature_dim=config['node_feature_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    gcn_trainer = GNNTrainer(gcn, config)
    gcn_trainer.train(train_data, val_data)
    gcn_metrics = gcn_trainer.evaluate(test_data)
    print(f"GCN Test PR-AUC: {gcn_metrics['PR_AUC']:.3f}")

    # 6. Train GAT
    print("\n--- Training GAT ---")
    gat = GATRiskModel(
        node_feature_dim=config['node_feature_dim'],
        hidden_dim=config['hidden_dim'],
        heads=config['heads'],
        out_dim=32,
        out_heads=config['out_heads'],
        dropout=config['dropout'],
        edge_dim=config['edge_feature_dim']
    )
    gat_trainer = GNNTrainer(gat, config)
    gat_trainer.train(train_data, val_data)
    gat_metrics = gat_trainer.evaluate(test_data)
    print(f"GAT Test PR-AUC: {gat_metrics['PR_AUC']:.3f}")

    # Combine metrics
    all_metrics = {
        'Logistic': baseline_metrics['Logistic'],
        'XGBoost': baseline_metrics['XGBoost'],
        'GCN': gcn_metrics,
        'GAT': gat_metrics
    }
    
    # 7. Disruption Stress Testing (using GAT)
    print("\n--- Running Network Disruption Analysis (GAT) ---")
    analyzer = DisruptionAnalyzer(gat, test_data, metadata)
    disruption_results = analyzer.run_stress_test(config['disruption_scenarios'])
    print(disruption_results[['Scenario', 'Global Risk', 'Downstream Delta']].to_string(index=False))

    # 8. Visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(all_metrics, OUT_DIR)
    plot_disruption_stress_test(disruption_results, OUT_DIR)
    plot_downstream_propagation(disruption_results, OUT_DIR)
    
    # Extract topology risk for one random test snapshot
    gat.eval()
    sample_data = test_data[0]
    with torch.no_grad():
        out = gat(sample_data.x, sample_data.edge_index, getattr(sample_data, 'edge_attr', None))
        probs = torch.sigmoid(out).cpu().numpy()
    plot_graph_topology_risk(metadata, probs, OUT_DIR)

    # 9. Save models and results
    print(f"\nSaving results to {OUT_DIR}/...")
    torch.save(gat.state_dict(), "models/gat_model.pt")
    torch.save(gcn.state_dict(), os.path.join(OUT_DIR, "gcn_model.pt"))
    
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(OUT_DIR, "phase5_metrics.csv"))
    
    print("\nPhase 5 execution complete.")


if __name__ == "__main__":
    main()
