import os
import yaml
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Any

class SupplyChainDataGenerator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']
        
        np.random.seed(self.config['seed'])
        self.start_date = pd.to_datetime(self.config['start_date'])
        self.end_date = pd.to_datetime(self.config['end_date'])
        self.dates = pd.date_range(self.start_date, self.end_date)
        
    def generate_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        plants = self._generate_plants()
        warehouses = self._generate_warehouses()
        regions = self._generate_regions()
        products = self._generate_products()
        
        # All locations for routing
        all_nodes = []
        for _, p in plants.iterrows():
            all_nodes.append({'id': p['plant_id'], 'type': 'plant', 'loc_x': p['loc_x'], 'loc_y': p['loc_y']})
        for _, w in warehouses.iterrows():
            all_nodes.append({'id': w['warehouse_id'], 'type': 'warehouse', 'loc_x': w['loc_x'], 'loc_y': w['loc_y']})
        for _, r in regions.iterrows():
            all_nodes.append({'id': r['region_id'], 'type': 'region', 'loc_x': r['loc_x'], 'loc_y': r['loc_y']})
        
        nodes_df = pd.DataFrame(all_nodes)
        
        routes = self._generate_routes(nodes_df)
        demand = self._generate_demand(regions, products)
        
        # Save to csv
        plants.to_csv(os.path.join(output_dir, 'plants.csv'), index=False)
        warehouses.to_csv(os.path.join(output_dir, 'warehouses.csv'), index=False)
        regions.to_csv(os.path.join(output_dir, 'regions.csv'), index=False)
        products.to_csv(os.path.join(output_dir, 'products.csv'), index=False)
        routes.to_csv(os.path.join(output_dir, 'routes.csv'), index=False)
        demand.to_csv(os.path.join(output_dir, 'demand.csv'), index=False)
        
        return {
            'plants': plants,
            'warehouses': warehouses,
            'regions': regions,
            'products': products,
            'routes': routes,
            'demand': demand
        }

    def _generate_plants(self) -> pd.DataFrame:
        n = self.config['n_plants']
        c = self.config['plant']
        data = []
        for i in range(n):
            cap = np.random.uniform(c['min_capacity'], c['max_capacity'])
            data.append({
                'plant_id': f'P{i+1}',
                'loc_x': np.random.uniform(0, 1000),
                'loc_y': np.random.uniform(0, 1000),
                'production_capacity': cap,
                'fixed_cost': np.random.uniform(c['min_fixed_cost'], c['max_fixed_cost']),
                'variable_production_cost': np.random.uniform(c['min_variable_cost'], c['max_variable_cost']),
                'energy_consumption': np.random.uniform(10, 50),
                'production_emission_factor': np.random.uniform(*c['emission_factor_range'])
            })
        return pd.DataFrame(data)

    def _generate_warehouses(self) -> pd.DataFrame:
        n = self.config['n_warehouses']
        c = self.config['warehouse']
        data = []
        for i in range(n):
            data.append({
                'warehouse_id': f'W{i+1}',
                'loc_x': np.random.uniform(0, 1000),
                'loc_y': np.random.uniform(0, 1000),
                'capacity': np.random.uniform(c['min_capacity'], c['max_capacity']),
                'holding_cost': np.random.uniform(c['min_holding_cost'], c['max_holding_cost'])
            })
        return pd.DataFrame(data)

    def _generate_regions(self) -> pd.DataFrame:
        n = self.config['n_regions']
        data = []
        for i in range(n):
            data.append({
                'region_id': f'R{i+1}',
                'loc_x': np.random.uniform(0, 1000),
                'loc_y': np.random.uniform(0, 1000),
                'base_demand_multiplier': np.random.uniform(0.5, 2.0)
            })
        return pd.DataFrame(data)

    def _generate_products(self) -> pd.DataFrame:
        n = self.config['n_products']
        families = ['A', 'B', 'C']
        data = []
        for i in range(n):
            data.append({
                'product_id': f'PRD{i+1}',
                'product_family': np.random.choice(families),
                'unit_weight': np.random.uniform(1.0, 20.0),
                'production_time': np.random.uniform(0.1, 2.0),
                'storage_requirement': np.random.uniform(1.0, 5.0),
                'safety_stock_requirement': np.random.uniform(100, 500)
            })
        return pd.DataFrame(data)

    def _generate_routes(self, nodes: pd.DataFrame) -> pd.DataFrame:
        c = self.config['transport']
        data = []
        
        # Plant -> Warehouse
        plants = nodes[nodes['type'] == 'plant']
        warehouses = nodes[nodes['type'] == 'warehouse']
        regions = nodes[nodes['type'] == 'region']
        
        for _, p in plants.iterrows():
            for _, w in warehouses.iterrows():
                dist = np.sqrt((p['loc_x'] - w['loc_x'])**2 + (p['loc_y'] - w['loc_y'])**2)
                data.append({
                    'source': p['id'],
                    'destination': w['id'],
                    'distance': dist,
                    'transport_cost': dist * c['cost_per_km_per_unit'],
                    'transport_time': np.ceil(dist / c['speed_km_per_day']),
                    'carbon_emission_factor': dist * c['emissions_per_km_per_unit'],
                    'capacity': np.random.uniform(10000, 50000)
                })
                
        # Warehouse -> Region
        for _, w in warehouses.iterrows():
            for _, r in regions.iterrows():
                dist = np.sqrt((w['loc_x'] - r['loc_x'])**2 + (w['loc_y'] - r['loc_y'])**2)
                # Not all warehouses serve all regions, say 50% probability
                if np.random.rand() > 0.5:
                    data.append({
                        'source': w['id'],
                        'destination': r['id'],
                        'distance': dist,
                        'transport_cost': dist * c['cost_per_km_per_unit'],
                        'transport_time': np.ceil(dist / c['speed_km_per_day']),
                        'carbon_emission_factor': dist * c['emissions_per_km_per_unit'],
                        'capacity': np.random.uniform(5000, 20000)
                    })
                    
        return pd.DataFrame(data)

    def _generate_demand(self, regions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        c = self.config['demand']
        data = []
        
        days_total = len(self.dates)
        time_index = np.arange(days_total)
        
        for _, r in regions.iterrows():
            base_mult = r['base_demand_multiplier']
            for _, p in products.iterrows():
                base_demand = np.random.uniform(c['base_min'], c['base_max']) * base_mult
                trend_factor = np.random.uniform(c['trend_range'][0], c['trend_range'][1])
                
                trend = base_demand * (1 + trend_factor * (time_index / 365.0))
                
                # Seasonality (weekly and yearly)
                yearly_seasonality = 1 + c['seasonality_amplitude'] * np.sin(2 * np.pi * time_index / 365.25)
                weekly_seasonality = 1 + (c['seasonality_amplitude']/2) * np.cos(2 * np.pi * time_index / 7)
                
                # Noise
                noise = np.random.normal(0, c['noise_level'] * base_demand, days_total)
                
                # Shocks
                shocks = np.zeros(days_total)
                shock_mask = np.random.rand(days_total) < c['shock_probability']
                shocks[shock_mask] = base_demand * np.random.uniform(c['shock_magnitude_range'][0], c['shock_magnitude_range'][1], sum(shock_mask))
                
                demand_series = np.maximum(0, trend * yearly_seasonality * weekly_seasonality + noise + shocks)
                
                df = pd.DataFrame({
                    'date': self.dates,
                    'region_id': r['region_id'],
                    'product_id': p['product_id'],
                    'demand': np.round(demand_series)
                })
                data.append(df)
                
        return pd.concat(data, ignore_index=True)

if __name__ == "__main__":
    generator = SupplyChainDataGenerator("configs/data.yaml")
    generator.generate_all("data/synthetic")
    print("Synthetic data generation complete.")
