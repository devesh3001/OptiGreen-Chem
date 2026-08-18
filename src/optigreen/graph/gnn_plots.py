"""
Visualization suite for Phase 5 GNN/GAT analysis.
Includes metrics comparison, disruption propagation, and topology heatmaps.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any


def plot_metrics_comparison(metrics: Dict[str, Dict[str, float]], save_dir: str):
    """Bar chart comparing PR-AUC and ROC-AUC across baselines and GNNs."""
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df, x='Model', y='PR_AUC', hue='Model', ax=ax1, palette='viridis', legend=False)
    ax1.set_title('Precision-Recall AUC (Primary Metric)')
    ax1.set_ylim(0, max(0.5, df['PR_AUC'].max() * 1.2))
    for i, v in enumerate(df['PR_AUC']):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
    sns.barplot(data=df, x='Model', y='ROC_AUC', hue='Model', ax=ax2, palette='mako', legend=False)
    ax2.set_title('ROC AUC')
    ax2.set_ylim(0.5, 1.0)
    for i, v in enumerate(df['ROC_AUC']):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gnn_metrics_comparison.png'), dpi=300)
    plt.close()


def plot_disruption_stress_test(results_df: pd.DataFrame, save_dir: str):
    """Waterfall or bar chart of risk increase under stress scenarios."""
    plt.figure(figsize=(10, 6))
    
    # Sort by impact
    df = results_df[results_df['Scenario'] != 'Baseline'].sort_values('Delta', ascending=True)
    
    bars = plt.barh(df['Scenario'], df['Delta'], color='indianred')
    plt.axvline(0, color='black', linewidth=1)
    
    plt.title('Global Network Risk Increase Under Disruption Scenarios', pad=20)
    plt.xlabel('Δ Mean Stockout Probability')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                 f"+{width:.3f}", va='center')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'disruption_stress_test.png'), dpi=300)
    plt.close()


def plot_downstream_propagation(results_df: pd.DataFrame, save_dir: str):
    """Compares global delta vs downstream-specific delta."""
    df = results_df[results_df['Scenario'] != 'Baseline'].copy()
    
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, df['Delta'], width, label='Global Risk Increase', color='lightgrey')
    ax.bar(x + width/2, df['Downstream Delta'], width, label='Downstream Risk Increase', color='crimson')
    
    ax.set_ylabel('Δ Stockout Probability')
    ax.set_title('Risk Propagation: Global vs Downstream Affected Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Scenario'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'risk_propagation.png'), dpi=300)
    plt.close()


def plot_graph_topology_risk(metadata: dict, node_risks: np.ndarray, save_dir: str):
    """
    Plots the supply chain network, coloring Region nodes by predicted risk.
    """
    # Build NetworkX graph from edge_index
    G = nx.DiGraph()
    all_ids = metadata['all_ids']
    id_to_idx = metadata['id_to_idx']
    
    for i, nid in enumerate(all_ids):
        node_type = 'plant' if nid.startswith('P') else 'warehouse' if nid.startswith('W') else 'region'
        G.add_node(nid, type=node_type, idx=i)
        
    edge_index = metadata['edge_index']  # expecting numpy array
    for i in range(edge_index.shape[1]):
        src = all_ids[edge_index[0, i]]
        dst = all_ids[edge_index[1, i]]
        G.add_edge(src, dst)
        
    # Layout (hierarchical: plants -> wh -> regions)
    pos = {}
    plants = [n for n, d in G.nodes(data=True) if d['type'] == 'plant']
    whs = [n for n, d in G.nodes(data=True) if d['type'] == 'warehouse']
    regions = [n for n, d in G.nodes(data=True) if d['type'] == 'region']
    
    for i, n in enumerate(plants): pos[n] = (i * 2, 2)
    for i, n in enumerate(whs): pos[n] = (i * 1.5, 1)
    for i, n in enumerate(regions): pos[n] = (i * 0.8, 0)
    
    plt.figure(figsize=(16, 10))
    
    # Draw plants (green) and WHs (blue)
    nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color='forestgreen', node_shape='s', node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=whs, node_color='steelblue', node_shape='o', node_size=400)
    
    # Draw regions colored by risk
    region_colors = []
    for r in regions:
        idx = id_to_idx[r]
        if idx < len(node_risks):
            region_colors.append(node_risks[idx])
        else:
            region_colors.append(0.0)
            
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=regions, node_color=region_colors, 
                                   cmap=plt.cm.Reds, vmin=0, vmax=1.0, node_shape='^', node_size=400)
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=False)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.colorbar(nodes, label='Predicted Stockout Probability')
    plt.title('Supply Chain Topology with GNN Risk Predictions')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graph_risk_topology.png'), dpi=300)
    plt.close()
