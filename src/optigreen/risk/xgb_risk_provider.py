"""
XGBoost Risk Provider for Phase 6 MILP integration.
Wraps the trained StockoutRiskModel to match the BaseRiskProvider interface.
"""
import pandas as pd
from typing import Dict, Tuple, List
from optigreen.risk.provider_interface import BaseRiskProvider
from optigreen.risk.risk_model import StockoutRiskModel


class XGBRiskProvider(BaseRiskProvider):
    def __init__(self,
                 model: StockoutRiskModel,
                 risk_features_df: pd.DataFrame):
        """
        Args:
            model: Trained StockoutRiskModel instance.
            risk_features_df: Pre-computed risk features DataFrame (e.g. from build_risk_features).
                              Must contain the features for the planning dates requested.
        """
        self.model = model
        self.risk_features_df = risk_features_df

    def get_risk_scores(self, planning_dates: List[str]) -> Dict[Tuple[str, str, int], float]:
        """
        Generates risk scores for specific dates using the XGBoost model.
        Returns a dict matching the optimizer's expected format.
        """
        # Get raw DataFrame predictions from the underlying model
        # We pass the full features df, the model filters to what it needs internally
        raw_scores_df = self.model.predict_risk_score(self.risk_features_df)
        
        # Filter to requested dates
        date_timestamps = pd.to_datetime(planning_dates)
        scores_for_dates = raw_scores_df[raw_scores_df['date'].isin(date_timestamps)]
        
        # Convert to dictionary formatted for MILP
        risk_dict = {}
        for day_idx, date in enumerate(planning_dates):
            day_ts = pd.Timestamp(date)
            day_data = scores_for_dates[scores_for_dates['date'] == day_ts]
            
            for _, row in day_data.iterrows():
                risk_dict[(row['region_id'], row['product_id'], day_idx)] = float(row['risk_score'])
                
        return risk_dict
