"""
ForecastProvider: Clean interface between ML forecasting and optimization engine.
Wraps trained ProbabilisticXGBoostForecaster to serve P10/P50/P90 per (region, product, date).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ForecastBundle:
    """Holds all quantile forecasts for a given slice."""
    region_id: str
    product_id: str
    date: pd.Timestamp
    p10: float
    p50: float
    p90: float

    @property
    def point(self) -> float:
        return self.p50

    @property
    def uncertainty_spread(self) -> float:
        """P90 - P10: width of the 80% prediction interval."""
        return self.p90 - self.p10


class ForecastProvider:
    """
    Decouples the forecasting model from the optimization engine.
    Any model that produces p10/p50/p90 can be plugged in by
    populating a forecast_df with the required columns.
    """

    def __init__(self, forecast_df: pd.DataFrame):
        """
        forecast_df must contain: date, region_id, product_id, P10, P50, P90
        """
        required_cols = {'date', 'region_id', 'product_id', 'P10', 'P50', 'P90'}
        missing = required_cols - set(forecast_df.columns)
        if missing:
            raise ValueError(f"forecast_df missing columns: {missing}")

        self._df = forecast_df.copy()
        self._df['date'] = pd.to_datetime(self._df['date'])
        self._index = self._df.set_index(['date', 'region_id', 'product_id'])

    def get(self, date, region_id: str, product_id: str) -> Optional[ForecastBundle]:
        """Retrieve a ForecastBundle for a specific (date, region, product)."""
        key = (pd.Timestamp(date), region_id, product_id)
        try:
            row = self._index.loc[key]
            return ForecastBundle(
                region_id=region_id,
                product_id=product_id,
                date=pd.Timestamp(date),
                p10=float(max(0, row['P10'])),
                p50=float(max(0, row['P50'])),
                p90=float(max(0, row['P90'])),
            )
        except KeyError:
            return None

    def get_horizon(self, start_date, horizon_days: int, region_id: str, product_id: str) -> pd.DataFrame:
        """
        Returns a DataFrame with forecasts for a (region, product) pair
        over [start_date, start_date + horizon_days).
        """
        dates = pd.date_range(start_date, periods=horizon_days, freq='D')
        rows = []
        for d in dates:
            b = self.get(d, region_id, product_id)
            if b is not None:
                rows.append({'date': b.date, 'region_id': b.region_id,
                             'product_id': b.product_id,
                             'P10': b.p10, 'P50': b.p50, 'P90': b.p90})
        return pd.DataFrame(rows)

    def get_all_for_dates(self, dates) -> pd.DataFrame:
        """Returns all forecasts for a list of dates."""
        dates = [pd.Timestamp(d) for d in dates]
        return self._df[self._df['date'].isin(dates)].copy()

    @property
    def available_dates(self):
        return sorted(self._df['date'].unique())

    @property
    def regions(self):
        return sorted(self._df['region_id'].unique())

    @property
    def products(self):
        return sorted(self._df['product_id'].unique())
