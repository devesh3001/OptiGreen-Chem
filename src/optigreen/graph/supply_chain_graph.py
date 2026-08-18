"""
Supply Chain Graph: NetworkX directed graph for Phase 5 GNN prep.
Nodes: Plants, Warehouses, Regions.
Edges: Plant→Warehouse routes + Warehouse→Region routes (synthesized from coordinates).
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any


def _euclidean_distance(x1, y1, x2, y2) -> float:
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def build_supply_chain_graph(
    plants_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    wh_region_cost_per_unit_km: float = 0.05,
    wh_region_carbon_per_unit_km: float = 0.02,
) -> nx.DiGraph:
    """
    Constructs a directed supply chain graph.

    Node types (stored in node attr 'node_type'):
        'plant', 'warehouse', 'region'

    Edge attributes:
        distance, transport_cost, capacity, transit_time, carbon_emission_factor

    Warehouse→Region edges are synthesized from Euclidean distance and
    configured per-unit-km costs (no real route data available for this link).
    """
    G = nx.DiGraph()

    # --- Add Plant nodes ---
    for _, row in plants_df.iterrows():
        G.add_node(row['plant_id'],
                   node_type='plant',
                   loc_x=row['loc_x'],
                   loc_y=row['loc_y'],
                   production_capacity=row['production_capacity'],
                   variable_production_cost=row['variable_production_cost'],
                   production_emission_factor=row['production_emission_factor'])

    # --- Add Warehouse nodes ---
    for _, row in warehouses_df.iterrows():
        G.add_node(row['warehouse_id'],
                   node_type='warehouse',
                   loc_x=row['loc_x'],
                   loc_y=row['loc_y'],
                   capacity=row['capacity'],
                   holding_cost=row['holding_cost'])

    # --- Add Region nodes ---
    for _, row in regions_df.iterrows():
        G.add_node(row['region_id'],
                   node_type='region',
                   loc_x=row['loc_x'],
                   loc_y=row['loc_y'],
                   base_demand_multiplier=row['base_demand_multiplier'])

    # --- Add Plant→Warehouse edges (from routes.csv) ---
    for _, row in routes_df.iterrows():
        G.add_edge(row['source'], row['destination'],
                   distance=row['distance'],
                   transport_cost=row['transport_cost'],
                   capacity=row['capacity'],
                   transit_time=row['transport_time'],
                   carbon_emission_factor=row['carbon_emission_factor'])

    # --- Synthesize Warehouse→Region edges ---
    for _, wrow in warehouses_df.iterrows():
        for _, rrow in regions_df.iterrows():
            dist = float(_euclidean_distance(
                wrow['loc_x'], wrow['loc_y'],
                rrow['loc_x'], rrow['loc_y']
            ))
            # Assume unlimited capacity for last-mile (can be bounded later)
            G.add_edge(wrow['warehouse_id'], rrow['region_id'],
                       distance=dist,
                       transport_cost=float(wh_region_cost_per_unit_km) * dist,
                       capacity=np.inf,
                       transit_time=max(1, int(dist / 200)),
                       carbon_emission_factor=float(wh_region_carbon_per_unit_km) * dist)

    return G


def graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Returns summary statistics for the supply chain graph."""
    plants = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'plant']
    warehouses = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'warehouse']
    regions = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'region']

    return {
        'n_plants': len(plants),
        'n_warehouses': len(warehouses),
        'n_regions': len(regions),
        'n_edges': G.number_of_edges(),
        'is_weakly_connected': nx.is_weakly_connected(G),
    }
