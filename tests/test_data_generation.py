import os
import pytest
import yaml
import pandas as pd
from optigreen.data.generator import SupplyChainDataGenerator
from optigreen.data.schemas import validate_datasets

@pytest.fixture
def config_file(tmp_path):
    config = {
        'data': {
            'seed': 42,
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'n_plants': 2,
            'n_warehouses': 3,
            'n_regions': 4,
            'n_products': 2,
            'plant': {
                'min_capacity': 100, 'max_capacity': 200,
                'min_fixed_cost': 10, 'max_fixed_cost': 20,
                'min_variable_cost': 1, 'max_variable_cost': 2,
                'emission_factor_range': [1, 2]
            },
            'warehouse': {
                'min_capacity': 50, 'max_capacity': 100,
                'min_holding_cost': 1, 'max_holding_cost': 2
            },
            'demand': {
                'base_min': 10, 'base_max': 20,
                'trend_range': [0, 0.01],
                'seasonality_amplitude': 0.1,
                'noise_level': 0.05,
                'shock_probability': 0,
                'shock_magnitude_range': [1, 2]
            },
            'transport': {
                'cost_per_km_per_unit': 0.1,
                'emissions_per_km_per_unit': 0.05,
                'speed_km_per_day': 100,
                'delay_probability': 0
            }
        }
    }
    config_path = tmp_path / "test_data.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return str(config_path)

def test_data_generation_and_validation(config_file, tmp_path):
    output_dir = tmp_path / "synthetic"
    generator = SupplyChainDataGenerator(config_file)
    data_dict = generator.generate_all(str(output_dir))
    
    assert len(data_dict['plants']) == 2
    assert len(data_dict['warehouses']) == 3
    assert len(data_dict['regions']) == 4
    assert len(data_dict['products']) == 2
    
    # 31 days * 4 regions * 2 products = 248 demand rows
    assert len(data_dict['demand']) == 248
    
    # Validate schemas
    validate_datasets(str(output_dir))
