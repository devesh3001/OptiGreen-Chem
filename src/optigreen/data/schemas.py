import pandera as pa
from pandera.typing import Series, DataFrame
import pandas as pd

class PlantSchema(pa.DataFrameModel):
    plant_id: Series[str]
    loc_x: Series[float] = pa.Field(ge=0, le=1000)
    loc_y: Series[float] = pa.Field(ge=0, le=1000)
    production_capacity: Series[float] = pa.Field(gt=0)
    fixed_cost: Series[float] = pa.Field(ge=0)
    variable_production_cost: Series[float] = pa.Field(gt=0)
    energy_consumption: Series[float] = pa.Field(gt=0)
    production_emission_factor: Series[float] = pa.Field(ge=0)

class WarehouseSchema(pa.DataFrameModel):
    warehouse_id: Series[str]
    loc_x: Series[float] = pa.Field(ge=0, le=1000)
    loc_y: Series[float] = pa.Field(ge=0, le=1000)
    capacity: Series[float] = pa.Field(gt=0)
    holding_cost: Series[float] = pa.Field(ge=0)

class RegionSchema(pa.DataFrameModel):
    region_id: Series[str]
    loc_x: Series[float] = pa.Field(ge=0, le=1000)
    loc_y: Series[float] = pa.Field(ge=0, le=1000)
    base_demand_multiplier: Series[float] = pa.Field(gt=0)

class ProductSchema(pa.DataFrameModel):
    product_id: Series[str]
    product_family: Series[str]
    unit_weight: Series[float] = pa.Field(gt=0)
    production_time: Series[float] = pa.Field(gt=0)
    storage_requirement: Series[float] = pa.Field(gt=0)
    safety_stock_requirement: Series[float] = pa.Field(ge=0)

class RouteSchema(pa.DataFrameModel):
    source: Series[str]
    destination: Series[str]
    distance: Series[float] = pa.Field(ge=0)
    transport_cost: Series[float] = pa.Field(ge=0)
    transport_time: Series[float] = pa.Field(ge=0)
    carbon_emission_factor: Series[float] = pa.Field(ge=0)
    capacity: Series[float] = pa.Field(gt=0)

class DemandSchema(pa.DataFrameModel):
    date: Series[pd.Timestamp]
    region_id: Series[str]
    product_id: Series[str]
    demand: Series[float] = pa.Field(ge=0)

def validate_datasets(data_dir: str):
    plants = pd.read_csv(f"{data_dir}/plants.csv")
    warehouses = pd.read_csv(f"{data_dir}/warehouses.csv")
    regions = pd.read_csv(f"{data_dir}/regions.csv")
    products = pd.read_csv(f"{data_dir}/products.csv")
    routes = pd.read_csv(f"{data_dir}/routes.csv")
    demand = pd.read_csv(f"{data_dir}/demand.csv", parse_dates=['date'])
    
    PlantSchema.validate(plants)
    WarehouseSchema.validate(warehouses)
    RegionSchema.validate(regions)
    ProductSchema.validate(products)
    RouteSchema.validate(routes)
    DemandSchema.validate(demand)
    print("All datasets validated successfully.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validate_datasets(sys.argv[1])
    else:
        validate_datasets("data/synthetic")
