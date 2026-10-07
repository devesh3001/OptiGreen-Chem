"""
Optimization result visualizations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Optional


plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


def plot_scenario_comparison(results_summary: pd.DataFrame, output_dir: str):
    """Bar chart comparison of key metrics across scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Optimization Scenario Comparison', fontsize=16, fontweight='bold')

    metrics = [
        ('Total Cost', 'Total Cost', 'Cost ($)'),
        ('Service Level', 'Service Level', 'Service Level'),
        ('Total Shortage', 'Total Shortage', 'Shortage (units)'),
        ('Avg Inventory', 'Avg Inventory', 'Avg Inventory (units)'),
        ('Total CO2', 'Total CO2', 'CO₂ Emissions'),
        ('Shortage Cost', 'Shortage Cost', 'Shortage Cost ($)'),
    ]

    for ax, (col, title, ylabel) in zip(axes.flatten(), metrics):
        scenarios = results_summary['Scenario'].tolist()
        values = results_summary[col].tolist()
        colors = PALETTE[:len(scenarios)]
        bars = ax.bar(range(len(scenarios)), values, color=colors)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scenario_comparison.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_pareto_frontier(pareto_df: pd.DataFrame, output_dir: str):
    """
    Cost vs CO₂ Pareto frontier as lambda_carbon is varied.
    pareto_df must have columns: lambda_carbon, total_cost, total_emissions
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(pareto_df['total_emissions'], pareto_df['total_cost'],
            'o-', color='#2196F3', linewidth=2, markersize=8)

    for _, row in pareto_df.iterrows():
        ax.annotate(f"λ={row['lambda_carbon']:.1f}",
                    (row['total_emissions'], row['total_cost']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Total CO₂ Emissions', fontsize=12)
    ax.set_ylabel('Total Cost ($)', fontsize=12)
    ax.set_title('Sustainability Pareto Frontier: Cost vs CO₂ Emissions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)

    path = os.path.join(output_dir, 'pareto_frontier.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_inventory_over_time(inventory_plan: pd.DataFrame, output_dir: str, scenario_name: str = ''):
    """Warehouse inventory levels over the planning horizon."""
    if inventory_plan is None or len(inventory_plan) == 0:
        return

    agg = inventory_plan.groupby('day')['inventory'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(agg['day'], agg['inventory'], alpha=0.3, color='#2196F3')
    ax.plot(agg['day'], agg['inventory'], 'o-', color='#2196F3', linewidth=2)
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Total Inventory (all warehouses)', fontsize=12)
    ax.set_title(f'Inventory Over Planning Horizon — {scenario_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)

    path = os.path.join(output_dir, f'inventory_{scenario_name.replace(" ", "_")}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_monte_carlo_results(mc_results: List, output_dir: str):
    """Box-plot style Monte Carlo cost distribution comparison."""
    names = [r.scenario_name for r in mc_results]
    means = [r.mean_cost for r in mc_results]
    stds = [r.std_cost for r in mc_results]
    p5s = [r.p5_cost for r in mc_results]
    p95s = [r.p95_cost for r in mc_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Monte Carlo Robustness Analysis (N=100 demand scenarios)', fontsize=14, fontweight='bold')

    # Cost distribution
    ax = axes[0]
    x = range(len(names))
    ax.bar(x, means, color=PALETTE[:len(names)], alpha=0.7, label='Mean Cost')
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=5, linewidth=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Mean ± Std Cost Under Demand Uncertainty')

    # Service level under uncertainty
    ax2 = axes[1]
    svc = [r.mean_service_level for r in mc_results]
    worst_svc = [r.worst_service_level for r in mc_results]
    ax2.bar(x, svc, color=PALETTE[:len(names)], alpha=0.7, label='Mean SL')
    ax2.scatter(x, worst_svc, marker='v', color='red', s=80, zorder=5, label='Worst-case SL')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Service Level')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.set_title('Service Level Under Demand Uncertainty')

    plt.tight_layout()
    path = os.path.join(output_dir, 'monte_carlo_robustness.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_risk_heatmap(risk_df: pd.DataFrame, output_dir: str):
    """Heatmap of risk scores across regions and products."""
    if risk_df is None or len(risk_df) == 0:
        return

    pivot = risk_df.groupby(['region_id', 'product_id'])['risk_score'].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)), max(6, len(pivot) // 2)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Avg Risk Score'})
    ax.set_title('ML Risk Score Heatmap: Region × Product', fontsize=13, fontweight='bold')
    ax.set_xlabel('Product')
    ax.set_ylabel('Region')

    path = os.path.join(output_dir, 'risk_heatmap.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_cost_breakdown(results_summary: pd.DataFrame, output_dir: str):
    """Stacked bar chart: cost breakdown per scenario."""
    cost_cols = ['Production Cost', 'Transport Cost', 'Holding Cost', 'Shortage Cost']
    available = [c for c in cost_cols if c in results_summary.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    scenarios = results_summary['Scenario'].tolist()
    x = np.arange(len(scenarios))
    bottoms = np.zeros(len(scenarios))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for col, color in zip(available, colors):
        vals = results_summary[col].fillna(0).values
        ax.bar(x, vals, bottom=bottoms, label=col, color=color, alpha=0.85)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Cost ($)')
    ax.set_title('Cost Breakdown by Scenario', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(output_dir, 'cost_breakdown.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
