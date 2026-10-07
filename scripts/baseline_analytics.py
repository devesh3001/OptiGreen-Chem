import pandas as pd
import numpy as np

def run_analytics(data_dir: str):
    print("=== OptiGreen-Chem Phase 1: Baseline Analytics ===")
    
    demand = pd.read_csv(f"{data_dir}/demand.csv", parse_dates=['date'])
    plants = pd.read_csv(f"{data_dir}/plants.csv")
    warehouses = pd.read_csv(f"{data_dir}/warehouses.csv")
    routes = pd.read_csv(f"{data_dir}/routes.csv")
    
    print(f"\n1. Topology:")
    print(f"   Plants: {len(plants)}")
    print(f"   Warehouses: {len(warehouses)}")
    print(f"   Routes: {len(routes)}")
    
    print(f"\n2. Plant Capacity vs Demand:")
    total_daily_capacity = plants['production_capacity'].sum()
    avg_daily_demand = demand.groupby('date')['demand'].sum().mean()
    max_daily_demand = demand.groupby('date')['demand'].sum().max()
    
    print(f"   Total Daily Plant Capacity: {total_daily_capacity:.0f}")
    print(f"   Average Daily System Demand: {avg_daily_demand:.0f}")
    print(f"   Peak Daily System Demand: {max_daily_demand:.0f}")
    
    if total_daily_capacity < avg_daily_demand:
        print("   WARNING: System is chronically under-capacitated!")
    elif total_daily_capacity < max_daily_demand:
        print("   INFO: System has capacity for average demand, but may struggle during peaks.")
    else:
        print("   INFO: System is over-capacitated.")
        
    print(f"\n3. Cost Profile:")
    avg_transport_cost = routes['transport_cost'].mean()
    print(f"   Average Transport Cost per Unit: {avg_transport_cost:.2f}")
    
    print(f"\n4. Demand Statistics:")
    print(f"   Total Data Points: {len(demand)}")
    print(f"   Min Demand: {demand['demand'].min()}")
    print(f"   Max Demand: {demand['demand'].max()}")
    
    print("\nAnalytics Complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_analytics(sys.argv[1])
    else:
        run_analytics("data/synthetic")
