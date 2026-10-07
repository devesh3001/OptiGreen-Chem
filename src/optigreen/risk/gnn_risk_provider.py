"""
GNN Risk Provider for Phase 4 MILP integration.
Wraps the trained GAT/GCN model and dataset builder to provide a generic
`get_risk_scores()` dictionary that the optimizer can consume without knowing
the underlying model architecture.
"""
import torch
import pandas as pd
from typing import Dict, Tuple, List
from optigreen.graph.graph_dataset import build_graph_snapshot
from optigreen.graph.gat_model import GATRiskModel
from optigreen.risk.provider_interface import BaseRiskProvider


class GNNRiskProvider(BaseRiskProvider):
    def __init__(self,
                 model: torch.nn.Module,
                 demand_df: pd.DataFrame,
                 plants_df: pd.DataFrame,
                 warehouses_df: pd.DataFrame,
                 regions_df: pd.DataFrame,
                 metadata: dict,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 weekly_features_df: pd.DataFrame):
        
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        self.demand_df = demand_df
        self.plants_df = plants_df
        self.warehouses_df = warehouses_df
        self.regions_df = regions_df
        self.metadata = metadata
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.weekly_df = weekly_features_df
        
        self.total_capacity = float(plants_df['production_capacity'].sum())

    @torch.no_grad()
    def get_risk_scores(self, planning_dates: List[str]) -> Dict[Tuple[str, str, int], float]:
        """
        Generate risk scores for a specific set of dates.
        We map dates to their corresponding weekly snapshot, run the GNN,
        and assign the resulting score to all days in that week.
        """
        risk_dict = {}
        
        dates = pd.to_datetime(planning_dates)
        weeks = pd.Series(dates.to_period('W')).apply(lambda p: p.start_time).unique()
        
        # Map Region nodes back to IDs
        id_to_idx = self.metadata['id_to_idx']
        all_ids = self.metadata['all_ids']
        n_plants = self.metadata['n_plants']
        n_wh = self.metadata['n_wh']
        products = self.metadata['products']
        product_to_idx = self.metadata['product_to_idx']
        
        for week in weeks:
            week_str = str(week.date())
            week_total_demand = float(self.weekly_df[
                self.weekly_df['week_start'] == week
            ]['weekly_demand'].sum())
            
            for product_id in products:
                # Build snapshot
                data = build_graph_snapshot(
                    week_start=week,
                    product_id=product_id,
                    product_idx=product_to_idx[product_id],
                    n_products=len(products),
                    weekly_df=self.weekly_df,
                    plants_df=self.plants_df,
                    warehouses_df=self.warehouses_df,
                    regions_df=self.regions_df,
                    all_ids=all_ids,
                    n_plants=n_plants,
                    n_wh=n_wh,
                    n_regions=self.metadata['n_regions'],
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr,
                    total_capacity=self.total_capacity,
                    total_weekly_demand=week_total_demand,
                )
                
                if data is None:
                    continue
                    
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, getattr(data, 'edge_attr', None))
                probs = torch.sigmoid(out).cpu().numpy()
                
                # Assign to region nodes
                for idx in range(n_plants + n_wh, len(all_ids)):
                    if data.y[idx] != -100:
                        region_id = all_ids[idx]
                        score = float(probs[idx])
                        
                        # Apply this score to all requested dates falling in this week
                        for day_idx, d in enumerate(dates):
                            if d.to_period('W').start_time == week:
                                risk_dict[(region_id, product_id, day_idx)] = score

        return risk_dict
