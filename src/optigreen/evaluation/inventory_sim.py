import pandas as pd
import numpy as np
from typing import Dict, Any

class InventorySimulation:
    def __init__(self, demand_df: pd.DataFrame, initial_inventory: float = 1000.0):
        """
        Simulate inventory decisions based on forecasts.
        demand_df must contain:
        - date, region_id, product_id, demand (actual)
        - predictions (e.g. naive, xgb_p50, prob_p90)
        """
        # Ensure chronological order
        self.df = demand_df.sort_values(by=['region_id', 'product_id', 'date']).reset_index(drop=True)
        self.initial_inventory = initial_inventory
        
        # Simple cost parameters
        self.holding_cost_per_unit = 2.0
        self.stockout_cost_per_unit = 10.0
        
    def simulate_strategy(self, forecast_col: str, safety_stock_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Strategy: At time t, we receive the forecast for t+1. 
        We order: forecast_t+1 * safety_stock_multiplier - current_inventory. 
        (Order arrives immediately for simplicity).
        Then actual demand happens.
        """
        total_holding_cost = 0.0
        total_stockout_cost = 0.0
        total_demand = 0.0
        total_stockout_units = 0.0
        
        results = []
        
        for (region, product), group in self.df.groupby(['region_id', 'product_id']):
            inventory = self.initial_inventory
            
            for _, row in group.iterrows():
                # The forecast for today
                forecast = max(0, row[forecast_col])
                actual_demand = row['demand']
                total_demand += actual_demand
                
                # We place order to reach target inventory level
                target_inventory = forecast * safety_stock_multiplier
                order_qty = max(0, target_inventory - inventory)
                
                # Inventory before demand
                inventory += order_qty
                
                # Demand happens
                if actual_demand > inventory:
                    stockout_qty = actual_demand - inventory
                    inventory = 0
                    total_stockout_units += stockout_qty
                    total_stockout_cost += stockout_qty * self.stockout_cost_per_unit
                else:
                    inventory -= actual_demand
                    
                # End of day holding
                total_holding_cost += inventory * self.holding_cost_per_unit
                
        total_cost = total_holding_cost + total_stockout_cost
        service_level = 1.0 - (total_stockout_units / total_demand) if total_demand > 0 else 1.0
        
        return {
            'Strategy': forecast_col,
            'Total Cost': total_cost,
            'Holding Cost': total_holding_cost,
            'Stockout Cost': total_stockout_cost,
            'Stockout Units': total_stockout_units,
            'Service Level': service_level
        }

    def run_all_strategies(self, strategies: Dict[str, str]) -> pd.DataFrame:
        """
        strategies: dict of 'Strategy Name' -> 'forecast_column'
        """
        results = []
        for name, col in strategies.items():
            # For P90, we might use safety_stock_multiplier = 1.0 because it's already conservative
            # For P50, we might need a multiplier if we want better service, but for direct comparison
            # let's just use 1.0 and rely on the forecast's own safety margin (i.e. P90 vs P50).
            res = self.simulate_strategy(col, safety_stock_multiplier=1.0)
            res['Strategy Name'] = name
            results.append(res)
            
        return pd.DataFrame(results)

from typing import List

def simulate_inventory_forward(
    actual_demand: pd.DataFrame,
    dates: List,
    regions: List,
    products: List,
    milp_shipments: pd.DataFrame,
    initial_inv: Dict,
    shortage_penalty: float = 200.0,
    holding_cost: float = 3.0
):
    total_shortage_qty = 0.0
    total_demand = 0.0
    region_shortages = {}
    
    print(f"DEBUG: milp_shipments columns = {milp_shipments.columns if milp_shipments is not None else 'None'}")
    if milp_shipments is not None and not milp_shipments.empty:
        print(milp_shipments.head())
    
    for day_idx, current_date in enumerate(dates):
        day_demand = actual_demand[actual_demand['date'] == pd.Timestamp(current_date)]
        
        if milp_shipments is None or milp_shipments.empty or 'day' not in milp_shipments.columns:
            day_shipments = pd.DataFrame(columns=['region', 'product', 'quantity'])
        else:
            day_shipments = milp_shipments[milp_shipments['day'] == day_idx]
        
        for r in regions:
            for p in products:
                incoming = day_shipments[
                    (day_shipments['region'] == r) & (day_shipments['product'] == p)
                ]['quantity'].sum()
                
                d = day_demand[
                    (day_demand['region_id'] == r) & (day_demand['product_id'] == p)
                ]['demand'].sum()
                
                # Debug info for the first day, first region/product
                if day_idx == 0 and r == regions[0] and p == products[0]:
                    print(f"Debug [Day {day_idx}, {r}, {p}]: incoming={incoming}, demand={d}")
                
                total_demand += d
                shortage = max(0.0, d - incoming)
                total_shortage_qty += shortage
                region_shortages[(r, p, current_date)] = shortage
                
    total_shortage_cost = total_shortage_qty * shortage_penalty
    service_level = max(0.0, 1.0 - (total_shortage_qty / max(total_demand, 1.0)))
    
    return {
        'total_shortage_qty': total_shortage_qty,
        'total_shortage_cost': total_shortage_cost,
        'total_holding_cost': 0.0,
        'service_level': service_level,
        'region_shortages': region_shortages
    }
