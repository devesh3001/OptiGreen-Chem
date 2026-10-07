"""Unit tests for risk feature construction and stockout risk model."""
import pytest
import pandas as pd
import numpy as np
from optigreen.risk.risk_features import build_risk_features
from optigreen.risk.risk_model import StockoutRiskModel, FEATURE_COLS
from optigreen.forecasting.forecast_provider import ForecastProvider, ForecastBundle


@pytest.fixture
def minimal_forecast_df():
    dates = pd.date_range('2024-01-01', periods=30)
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for region in ['R1', 'R2']:
            for product in ['PRD1']:
                p50 = rng.uniform(80, 120)
                spread = rng.uniform(5, 30)
                rows.append({
                    'date': d, 'region_id': region, 'product_id': product,
                    'P10': max(0, p50 - spread), 'P50': p50, 'P90': p50 + spread,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_demand_df(minimal_forecast_df):
    rng = np.random.default_rng(99)
    rows = []
    for _, row in minimal_forecast_df.iterrows():
        noise = rng.normal(0, 15)
        rows.append({
            'date': row['date'], 'region_id': row['region_id'],
            'product_id': row['product_id'],
            'demand': max(0, row['P50'] + noise),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_plants_df():
    return pd.DataFrame([
        {'plant_id': 'P1', 'production_capacity': 5000.0},
        {'plant_id': 'P2', 'production_capacity': 5000.0},
    ])


class TestForecastProvider:
    def test_get_returns_correct_quantiles(self, minimal_forecast_df):
        provider = ForecastProvider(minimal_forecast_df)
        bundle = provider.get('2024-01-05', 'R1', 'PRD1')
        assert bundle is not None
        assert isinstance(bundle, ForecastBundle)
        assert bundle.p10 <= bundle.p50 <= bundle.p90

    def test_missing_key_returns_none(self, minimal_forecast_df):
        provider = ForecastProvider(minimal_forecast_df)
        result = provider.get('2024-01-01', 'NONEXISTENT', 'PRD1')
        assert result is None

    def test_uncertainty_spread_positive(self, minimal_forecast_df):
        provider = ForecastProvider(minimal_forecast_df)
        bundle = provider.get('2024-01-10', 'R1', 'PRD1')
        assert bundle.uncertainty_spread >= 0.0

    def test_regions_and_products(self, minimal_forecast_df):
        provider = ForecastProvider(minimal_forecast_df)
        assert 'R1' in provider.regions
        assert 'PRD1' in provider.products


class TestRiskFeatures:
    def test_build_risk_features_shape(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        assert len(features) > 0
        assert 'forecast_spread' in features.columns
        assert 'relative_spread' in features.columns
        assert 'stockout_label' in features.columns
        assert 'capacity_utilization' in features.columns

    def test_risk_features_no_negative_spread(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        assert (features['forecast_spread'] >= 0).all(), "Spread must be >= 0"

    def test_stockout_label_binary(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        unique_labels = features['stockout_label'].unique()
        assert set(unique_labels).issubset({0, 1}), f"Labels must be 0 or 1, got {unique_labels}"


class TestStockoutRiskModel:
    def test_risk_score_range(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        """Risk scores must be in [0, 1]."""
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        model = StockoutRiskModel()
        metrics = model.train(features)
        risk_df = model.predict_risk_score(features)

        assert 'risk_score' in risk_df.columns
        assert (risk_df['risk_score'] >= 0.0).all()
        assert (risk_df['risk_score'] <= 1.0).all()

    def test_metrics_returned(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        """Training must return a dict with standard evaluation metrics."""
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        model = StockoutRiskModel()
        metrics = model.train(features)

        assert 'ROC_AUC' in metrics
        assert 'PR_AUC' in metrics
        assert 'F1' in metrics

    def test_statistical_fallback_risk_score(self, minimal_forecast_df):
        """Statistical fallback should produce risk scores without a trained classifier."""
        model = StockoutRiskModel()
        result = model.compute_statistical_risk_score(minimal_forecast_df)
        assert 'risk_score' in result.columns
        assert (result['risk_score'] >= 0.0).all()
        assert (result['risk_score'] <= 1.0).all()

    def test_untrained_model_raises(self, minimal_forecast_df, minimal_demand_df, minimal_plants_df):
        """Calling predict on untrained model should raise ValueError."""
        features = build_risk_features(minimal_forecast_df, minimal_demand_df, minimal_plants_df)
        model = StockoutRiskModel()
        with pytest.raises(ValueError, match="not trained"):
            model.predict_risk_score(features)
