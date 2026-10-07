import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_actual_vs_predicted(df: pd.DataFrame, title: str, output_path: str, region_id: str = None, product_id: str = None):
    """
    Plots Actual vs Predicted demand over time.
    Optionally filters by region and product.
    """
    plot_df = df.copy()
    if region_id:
        plot_df = plot_df[plot_df['region_id'] == region_id]
    if product_id:
        plot_df = plot_df[plot_df['product_id'] == product_id]
        
    # Aggregate by date if multiple regions/products are present
    agg_df = plot_df.groupby('date')[['demand', 'prediction']].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(agg_df['date'], agg_df['demand'], label='Actual Demand', color='blue', alpha=0.7)
    plt.plot(agg_df['date'], agg_df['prediction'], label='Predicted Demand', color='orange', alpha=0.7)
    
    title_suffix = ""
    if region_id or product_id:
        title_suffix = f" (Region: {region_id or 'All'}, Product: {product_id or 'All'})"
        
    plt.title(title + title_suffix)
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_error_distribution(df: pd.DataFrame, title: str, output_path: str):
    """
    Plots the distribution of prediction errors (Actual - Predicted)
    """
    errors = df['demand'] - df['prediction']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True)
    plt.title(title + " - Error Distribution (Actual - Predicted)")
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
