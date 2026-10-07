"""
Network Propagation Analysis: Controlled disruption stress testing.
Modifies node/edge features in-place on test snapshots and measures
how GNN risk predictions change downstream.
"""
import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List


class DisruptionAnalyzer:
    def __init__(self, model, test_data: list, metadata: dict):
        self.model = model
        self.model.eval()
        self.base_data = test_data
        self.meta = metadata
        self.device = next(model.parameters()).device
        
        self.id_to_idx = metadata['id_to_idx']
        self.all_ids = metadata['all_ids']
        self.n_plants = metadata['n_plants']
        self.n_wh = metadata['n_wh']

    @torch.no_grad()
    def _predict_all(self, data_list: list) -> Dict[str, float]:
        """Returns mean predicted risk across all region nodes."""
        all_probs = []
        for d in data_list:
            d = d.to(self.device)
            out = self.model(d.x, d.edge_index, getattr(d, 'edge_attr', None))
            mask = d.y != -100
            if mask.any():
                probs = torch.sigmoid(out[mask]).cpu().numpy()
                all_probs.extend(probs)
        return float(np.mean(all_probs)) if all_probs else 0.0

    @torch.no_grad()
    def _predict_downstream(self, data_list: list, target_region_ids: List[str]) -> float:
        """Returns mean risk for specific region nodes."""
        target_indices = [self.id_to_idx[rid] for rid in target_region_ids if rid in self.id_to_idx]
        if not target_indices:
            return 0.0
            
        all_probs = []
        for d in data_list:
            d = d.to(self.device)
            out = self.model(d.x, d.edge_index, getattr(d, 'edge_attr', None))
            
            # Mask to target indices
            mask = torch.zeros(len(d.y), dtype=torch.bool, device=self.device)
            for idx in target_indices:
                if d.y[idx] != -100:  # safety check
                    mask[idx] = True
                    
            if mask.any():
                probs = torch.sigmoid(out[mask]).cpu().numpy()
                all_probs.extend(probs)
        return float(np.mean(all_probs)) if all_probs else 0.0

    def get_downstream_regions(self, source_id: str) -> List[str]:
        """Finds regions downstream of a plant or warehouse."""
        regions = []
        edge_index = self.base_data[0].edge_index.cpu().numpy()
        src_idx = self.id_to_idx.get(source_id)
        if src_idx is None:
            return []
            
        # Very simple 2-hop traversal (Plant -> WH -> Region)
        connected_whs = []
        for i in range(edge_index.shape[1]):
            if edge_index[0, i] == src_idx:
                dst = edge_index[1, i]
                if self.all_ids[dst].startswith('W'):
                    connected_whs.append(dst)
                elif self.all_ids[dst].startswith('R'):
                    regions.append(self.all_ids[dst])
                    
        for wh_idx in connected_whs:
            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == wh_idx:
                    dst = edge_index[1, i]
                    if self.all_ids[dst].startswith('R'):
                        regions.append(self.all_ids[dst])
                        
        return list(set(regions))

    def run_stress_test(self, scenario_cfg: dict) -> pd.DataFrame:
        """
        Runs multiple disruption scenarios defined in config.
        """
        base_risk = self._predict_all(self.base_data)
        results = [{'Scenario': 'Baseline', 'Global Risk': base_risk, 'Delta': 0.0, 'Downstream Delta': 0.0}]
        
        for name, params in scenario_cfg.items():
            mod_data = [copy.deepcopy(d) for d in self.base_data]
            downstream_regions = []
            
            if 'plant_id' in params and 'capacity_fraction' in params:
                pid = params['plant_id']
                idx = self.id_to_idx.get(pid)
                if idx is not None:
                    # Feature 0 is capacity_norm. We just multiply by fraction.
                    for d in mod_data:
                        d.x[idx, 0] *= params['capacity_fraction']
                    downstream_regions = self.get_downstream_regions(pid)
                    
            elif 'warehouse_id' in params and 'capacity_fraction' in params:
                wid = params['warehouse_id']
                idx = self.id_to_idx.get(wid)
                if idx is not None:
                    # Feature 0 is capacity_norm for WH
                    for d in mod_data:
                        d.x[idx, 0] *= params['capacity_fraction']
                    downstream_regions = self.get_downstream_regions(wid)
                    
            elif 'regions' in params and 'spike_multiplier' in params:
                for rid in params['regions']:
                    idx = self.id_to_idx.get(rid)
                    if idx is not None:
                        # Feature 0 is p50 proxy for Region
                        for d in mod_data:
                            d.x[idx, 0] *= params['spike_multiplier']
                            d.x[idx, 1] *= params['spike_multiplier'] # spread
                downstream_regions = params['regions']
                
            elif 'combined' in params:
                # Apply hardcoded combo for test
                pid = "P1"
                idx = self.id_to_idx.get(pid)
                if idx is not None:
                    for d in mod_data:
                        d.x[idx, 0] *= params['plant_capacity_fraction']
                downstream_regions = self.get_downstream_regions(pid)

            # Re-predict
            new_global = self._predict_all(mod_data)
            
            if downstream_regions:
                base_downstream = self._predict_downstream(self.base_data, downstream_regions)
                new_downstream = self._predict_downstream(mod_data, downstream_regions)
                downstream_delta = new_downstream - base_downstream
            else:
                downstream_delta = 0.0
                
            results.append({
                'Scenario': name,
                'Global Risk': new_global,
                'Delta': new_global - base_risk,
                'Downstream Delta': downstream_delta
            })
            
        return pd.DataFrame(results)
