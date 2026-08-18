"""
Risk Provider Interface
Abstracts the source of risk scores from the MILP optimizer.
"""
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod


class BaseRiskProvider(ABC):
    """
    Abstract interface for all supply chain risk intelligence providers.
    The optimizer depends strictly on this interface.
    """
    
    @abstractmethod
    def get_risk_scores(self, planning_dates: List[str]) -> Dict[Tuple[str, str, int], float]:
        """
        Returns a dictionary of risk scores for the requested planning dates.
        
        Args:
            planning_dates: List of date strings (e.g. ['2025-07-24', '2025-07-25', ...])
            
        Returns:
            Dict mapping (region_id, product_id, day_idx) to a risk score in [0, 1].
            day_idx corresponds to the index of the date in planning_dates.
        """
        pass
