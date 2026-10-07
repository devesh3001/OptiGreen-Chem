"""
Physical network for the OptiGreen-Chem real-data case (U.S. refined products).

Nodes
-----
* 12 plants   = EIA refining districts (real capacity, utilisation, yields).
* 10 hubs     = bulk-terminal systems. Seven map 1:1 to EIA stock areas
                (1A, 1B, P3, P4 ...). Three EIA areas are split in two because
                the area is physically served from two directions:
                1C -> Atlanta (pipeline) + Tampa (marine, Florida)
                P2 -> Chicago (north) + Tulsa (south)
                P5 -> Los Angeles + Seattle (Pacific Northwest)
                Stocks of a split area are divided by trailing demand share
                (documented in realcase.split_stocks).
* regions     = 50 states + DC, plus one "withheld remainder" pool per EIA
                demand area (EIA withholds some state values; the pool is the
                published area total minus the published states).

Coordinates are approximate population / refinery-cluster centres (decimal
degrees, accurate to roughly +/-50 km). Distances are great-circle distances
multiplied by a circuity factor (road/pipeline 1.25; sea lanes use tabulated
port-to-port distances).

Emission factors (g CO2 per tonne-km) for chemical-industry freight are the
Cefic/ECTA "Guidelines for measuring and managing CO2 emission from freight
transport operations" (Issue 1, 2011), Table 10 (A. McKinnon):
    road 62, rail 22, barge 31, short-sea 16, deep-sea tanker 5, pipeline 5.

Tariff anchor for product pipelines: Colonial Pipeline FERC No. 99.95.0
(effective 1 Sep 2026): Houston->Linden 351.91 cents/bbl, Houston->Atlanta
120.65 cents/bbl, i.e. about $0.009-0.012 per tonne-km for gasoline.
Other unit costs are stated planning assumptions (see COST_ASSUMPTIONS).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
# state -> (EIA demand area, lat, lon)
STATE_INFO: Dict[str, Tuple[str, float, float]] = {
    "CT": ("1A", 41.50, -72.90), "ME": ("1A", 44.10, -69.90), "MA": ("1A", 42.30, -71.40),
    "NH": ("1A", 43.00, -71.50), "RI": ("1A", 41.75, -71.45), "VT": ("1A", 44.10, -72.80),
    "DE": ("1B", 39.40, -75.60), "DC": ("1B", 38.90, -77.02), "MD": ("1B", 39.10, -76.80),
    "NJ": ("1B", 40.40, -74.40), "NY": ("1B", 41.50, -74.60), "PA": ("1B", 40.50, -77.00),
    "FL": ("1C", 27.80, -81.60), "GA": ("1C", 33.30, -84.00), "NC": ("1C", 35.50, -79.60),
    "SC": ("1C", 34.00, -81.00), "VA": ("1C", 37.90, -77.80), "WV": ("1C", 38.80, -81.10),
    "IL": ("P2", 41.30, -88.40), "IN": ("P2", 40.00, -86.30), "IA": ("P2", 41.90, -93.00),
    "KS": ("P2", 38.45, -96.80), "KY": ("P2", 37.80, -85.50), "MI": ("P2", 42.90, -84.20),
    "MN": ("P2", 45.30, -93.60), "MO": ("P2", 38.40, -92.20), "NE": ("P2", 41.00, -97.40),
    "ND": ("P2", 47.40, -99.00), "SD": ("P2", 44.00, -98.90), "OH": ("P2", 40.50, -82.60),
    "OK": ("P2", 35.60, -96.80), "TN": ("P2", 35.80, -86.40), "WI": ("P2", 43.70, -89.00),
    "AL": ("P3", 33.00, -86.70), "AR": ("P3", 35.10, -92.40), "LA": ("P3", 30.70, -91.40),
    "MS": ("P3", 32.60, -89.60), "NM": ("P3", 34.60, -106.30), "TX": ("P3", 30.90, -97.40),
    "CO": ("P4", 39.50, -105.20), "ID": ("P4", 43.60, -115.40), "MT": ("P4", 46.50, -110.80),
    "UT": ("P4", 40.40, -111.90), "WY": ("P4", 42.80, -106.90),
    "AK": ("P5", 61.40, -149.00), "AZ": ("P5", 33.40, -111.90), "CA": ("P5", 35.50, -119.40),
    "HI": ("P5", 21.30, -157.85), "NV": ("P5", 36.50, -115.40), "OR": ("P5", 44.70, -122.60),
    "WA": ("P5", 47.30, -121.60),
}
DEMAND_AREAS = ["1A", "1B", "1C", "P2", "P3", "P4", "P5"]

# hub -> (stock area, lat, lon, is_port, label)
HUBS: Dict[str, Tuple[str, float, float, bool, str]] = {
    "H1A": ("1A", 42.36, -71.06, True, "Boston (New England)"),
    "H1B": ("1B", 40.63, -74.24, True, "New York Harbor / Linden"),
    "H1C": ("1C", 33.75, -84.39, False, "Atlanta (Colonial/Plantation)"),
    "H1F": ("1C", 27.95, -82.45, True, "Tampa (Florida, marine)"),
    "H2N": ("P2", 41.70, -87.70, False, "Chicago (Midwest north)"),
    "H2S": ("P2", 36.15, -95.99, False, "Tulsa (Midwest south)"),
    "H3": ("P3", 29.75, -95.25, True, "Houston (Gulf Coast)"),
    "H4": ("P4", 39.80, -104.95, False, "Denver (Rocky Mountains)"),
    "H5": ("P5", 33.77, -118.25, True, "Los Angeles (West Coast south)"),
    "H5N": ("P5", 47.60, -122.35, True, "Seattle / Puget Sound"),
}
# split areas: which states each sub-hub serves first (others by distance)
HUB_PRIMARY_STATES = {
    "H1F": ["FL"],
    "H5N": ["WA", "OR", "AK"],
}
# States with no interstate product pipeline: they are supplied through their own marine
# terminals (EIA State Energy Profile, Florida: products "arrive by tanker and barge").
# Any other hub can reach them only by road tanker, never at pipeline rates.
MARINE_SUPPLIED_STATES = {"FL"}

# refining district -> (PADD, lat, lon, label)
DISTRICT_INFO: Dict[str, Tuple[str, float, float, str]] = {
    "EC": ("P1", 39.85, -75.25, "East Coast (Delaware Valley)"),
    "AP": ("P1", 40.40, -79.90, "Appalachian No. 1 (W. PA / WV)"),
    "2A": ("P2", 40.60, -87.50, "Indiana-Illinois-Kentucky"),
    "2B": ("P2", 44.80, -93.00, "Minnesota-Wisconsin-Dakotas"),
    "2C": ("P2", 36.80, -96.80, "Oklahoma-Kansas-Missouri"),
    "3A": ("P3", 32.50, -101.00, "Texas Inland"),
    "3B": ("P3", 29.40, -94.70, "Texas Gulf Coast"),
    "3C": ("P3", 30.20, -91.30, "Louisiana Gulf Coast"),
    "3D": ("P3", 32.80, -93.30, "North Louisiana-Arkansas"),
    "3E": ("P3", 32.80, -104.40, "New Mexico"),
    "P4": ("P4", 42.00, -106.50, "Rocky Mountain"),
    "P5": ("P5", 36.50, -120.50, "West Coast"),
}
COASTAL_DISTRICTS = {"EC", "3B", "3C", "P5"}  # exposed to hurricanes / marine access
MARINE_ORIGIN_DISTRICTS = {"EC", "3B", "3C", "P5"}  # load tankers at their own ports
RIVER_ORIGIN_DISTRICTS = {"2A", "2B", "2C", "3B", "3C", "3D"}  # barge access to the Mississippi system

# EIA movement area -> hubs that receive it, by mode
MOVEMENT_DEST_HUBS = {
    ("P1", "pipeline"): ["H1B", "H1C"],
    ("1A", "marine"): ["H1A"],
    ("1B", "marine"): ["H1B"],
    ("1C", "marine"): ["H1F"],
    ("P2", "pipeline"): ["H2N", "H2S"],
    ("P2", "marine"): ["H2N", "H2S"],
    ("P3", "pipeline"): ["H3"],
    ("P3", "marine"): ["H3"],
    ("P4", "pipeline"): ["H4"],
    ("P5", "pipeline"): ["H5"],
    ("P5", "marine"): ["H5"],
}
# approximate sea-lane distances, km (port of loading -> hub port)
SEA_KM = {
    ("P3", "H1A"): 3900, ("P3", "H1B"): 3520, ("P3", "H1F"): 1600, ("P3", "H5"): 8700,
    ("P1", "H1A"): 650, ("P5", "H5N"): 1900, ("P5", "H5"): 600,
    ("H5", "HI"): 4130, ("H5N", "AK"): 2700,
    ("IMPORT", "H1A"): 5600, ("IMPORT", "H1B"): 6300, ("IMPORT", "H1F"): 7400,
    ("IMPORT", "H3"): 8900, ("IMPORT", "H5"): 9500, ("IMPORT", "H5N"): 8000,
}
# river distance for inland barge (Gulf <-> Midwest, Mississippi/Ohio system)
RIVER_KM = {("P3", "P2"): 1700, ("P2", "P3"): 1700, ("P2", "P1"): 1500, ("P1", "P2"): 1500}
IMPORT_HUBS = ["H1A", "H1B", "H1F", "H3", "H5", "H5N"]
EXPORT_HUBS = {"P3": "H3", "P5": "H5", "P1": "H1B", "P2": "H2N", "P4": "H4"}

# --------------------------------------------------------------------------- #
# Emission and cost factors
# --------------------------------------------------------------------------- #
EMISSION_G_PER_TKM = {  # Cefic/ECTA 2011, Table 10
    "road": 62.0, "rail": 22.0, "barge": 31.0, "short_sea": 16.0,
    "deep_sea_tanker": 5.0, "pipeline": 5.0,
}
COST_ASSUMPTIONS = {
    # $/t-km; pipeline anchored on the Colonial 2026 FERC tariff (see module doc)
    "pipeline_usd_per_tkm": 0.010,
    "truck_usd_per_tkm": 0.10,          # tank-truck, ~$3.5 per loaded mile for 25 t
    "truck_fixed_usd_per_t": 3.0,       # rack loading / unloading
    "last_mile_km": 60.0,               # terminal rack -> customer, by truck
    # marine voyages (lump-sum per cargo; integer decision in the MILP)
    "tanker_cargo_kbbl": 300.0,         # MR product tanker
    "tanker_usd_per_day": 95_000.0,     # Jones Act MR time-charter + bunkers (assumption)
    "tanker_speed_km_per_day": 578.0,   # 13 knots
    "tanker_port_days": 4.0,
    "barge_tow_kbbl": 75.0,             # 3-barge tow on the Mississippi/Ohio
    "barge_usd_per_bbl_per_1000km": 0.9,
    "refining_opex_usd_per_bbl": 5.0,   # variable opex excluding crude (assumption)
    "holding_usd_per_bbl_month": 0.50,  # terminal storage lease rate (assumption)
    "shortage_multiplier": 2.0,         # unmet demand penalty = 2 x product spot value
    "export_shortfall_multiplier": 1.5, # missed export cargo = replacement at spot + 50 % (contract penalty)
    "stock_target_multiplier": 1.25,    # ending stock below seasonal target, valued at 1.25 x spot
    "emergency_import_premium": 0.25,   # recourse imports at spot x 1.25
}
ETHANOL_BLEND_NOTE = (
    "EIA refinery yield of finished motor gasoline excludes ethanol and NGL blendstocks; "
    "finished-gasoline supply = crude x yield x blend uplift, with the uplift estimated "
    "from EIA data outside PADD 1 (realcase.gasoline_blend_uplift)."
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def road_km(a: Tuple[float, float], b: Tuple[float, float], circuity: float = 1.25) -> float:
    return circuity * haversine_km(a[0], a[1], b[0], b[1])


def hub_coord(h: str) -> Tuple[float, float]:
    return HUBS[h][1], HUBS[h][2]


def district_coord(d: str) -> Tuple[float, float]:
    return DISTRICT_INFO[d][1], DISTRICT_INFO[d][2]


def state_coord(s: str) -> Tuple[float, float]:
    return STATE_INFO[s][1], STATE_INFO[s][2]


def hubs_of_area(area: str) -> List[str]:
    return [h for h, v in HUBS.items() if v[0] == area]


def tanker_voyage_usd(distance_km: float) -> float:
    c = COST_ASSUMPTIONS
    days = 2 * distance_km / c["tanker_speed_km_per_day"] + c["tanker_port_days"]
    return days * c["tanker_usd_per_day"]


def barge_tow_usd(distance_km: float) -> float:
    c = COST_ASSUMPTIONS
    return c["barge_usd_per_bbl_per_1000km"] * c["barge_tow_kbbl"] * 1000 * distance_km / 1000.0


def primary_hub_for_state(state: str) -> str:
    for h, states in HUB_PRIMARY_STATES.items():
        if state in states:
            return h
    area = STATE_INFO[state][0]
    cands = [h for h in hubs_of_area(area) if h not in HUB_PRIMARY_STATES]
    return min(cands, key=lambda h: road_km(state_coord(state), hub_coord(h)))
