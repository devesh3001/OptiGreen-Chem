import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_disruption_strategy_comparison(results_df: pd.DataFrame, out_dir: str):
    """Plot average Cost and Service Level across strategies for each scenario."""
    plt.figure(figsize=(12, 6))
    
    # Plot Cost
    plt.subplot(1, 2, 1)
    sns.barplot(data=results_df, x='Scenario', y='Total Cost', hue='Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.title('Total Cost by Scenario and Strategy')
    plt.tight_layout()
    
    # Plot Service Level
    plt.subplot(1, 2, 2)
    sns.barplot(data=results_df, x='Scenario', y='Service Level', hue='Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.title('Service Level by Scenario and Strategy')
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir, 'scenario_comparison.png'))
    plt.close()

def plot_phase6_pareto_frontier(results_df: pd.DataFrame, out_dir: str):
    """Plot Cost vs CO2 (Sustainability) and Cost vs Service Level (Efficiency)."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=results_df, x='CO2', y='Total Cost', hue='Strategy', style='Scenario', s=100)
    plt.title('Sustainability: Cost vs CO2')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=results_df, x='Service Level', y='Total Cost', hue='Strategy', style='Scenario', s=100)
    plt.title('Efficiency: Cost vs Service Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_frontier.png'))
    plt.close()

def plot_monte_carlo_risk_comparison(mc_results: list, out_dir: str):
    """Plot Monte Carlo Cost and Service Level distribution summary."""
    mc_df = pd.DataFrame(mc_results)
    if mc_df.empty: return
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=mc_df, x='scenario_name', y='mean_cost')
    plt.xticks(rotation=45, ha='right')
    plt.title('Monte Carlo: Mean Cost')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=mc_df, x='scenario_name', y='mean_service_level')
    plt.xticks(rotation=45, ha='right')
    plt.title('Monte Carlo: Mean Service Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'monte_carlo_comparison.png'))
    plt.close()
