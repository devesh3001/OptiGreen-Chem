"""
EIA petroleum bulk-file reader for OptiGreen-Chem (real-data case).

Source
------
U.S. Energy Information Administration, "Petroleum and other liquid fuels"
API bulk file PET.zip  ->  https://www.eia.gov/opendata/bulk/PET.zip
(public domain, no API key needed). The file is newline-delimited JSON; every
line is one time series with `series_id`, `name`, `units`, `f` (frequency) and
`data` = [[period, value], ...].

This module streams the file once, keeps only the series that OptiGreen-Chem
needs (matched by their published *names*, so the selection is auditable), and
writes small tidy CSV tables to `data/real/eia/`.

Tables produced
---------------
demand_state_monthly.csv   month, state, product, kgal_per_day, kt
district_monthly.csv       month, district, capacity_kbcd, utilization_pct,
                           crude_input_kbd, prod_<K>_kbbl (5 products)
stocks_monthly.csv         month, area, product, kbbl
movements_monthly.csv      month, from_area, to_area, mode, product, kbbl
trade_monthly.csv          month, padd, flow (imports/exports), product, kbbl
prices_monthly.csv         month, series, usd (per gal or per bbl, see `unit`)
refinery_fuel_annual.csv   year, padd, fuel, value, units
utilization_weekly.csv     week, padd, utilization_pct (2010+)
series_used.csv            audit list of every EIA series id that was read

Units
-----
Prime-supplier sales are reported in thousand gallons per day (monthly mean).
Production, stocks and movements are in thousand barrels (monthly totals).
Mass conversions use product densities at 15 degC (see DENSITY_T_PER_M3).
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Products and physical constants
# --------------------------------------------------------------------------- #
PRODUCTS = ["GAS", "DIS", "JET", "RES", "LPG"]
PRODUCT_NAMES = {
    "GAS": "Motor gasoline",
    "DIS": "Distillate fuel oil (diesel/heating oil)",
    "JET": "Kerosene-type jet fuel",
    "RES": "Residual fuel oil",
    "LPG": "Propane / propylene",
}
# Typical densities at 15 degC, t/m3 (ASTM/API typical values; propane as liquid).
DENSITY_T_PER_M3 = {"GAS": 0.745, "DIS": 0.845, "JET": 0.800, "RES": 0.980, "LPG": 0.508}
M3_PER_BBL = 0.158987295
M3_PER_KGAL = 3.785411784
CRUDE_DENSITY_T_PER_M3 = 0.870  # ~31 API average U.S. refinery crude slate

# Name fragments used by EIA for each product, per table
PRIME_SUPPLIER_PRODUCT = {
    "Total Gasoline": "GAS",
    "Total Distillate plus Kerosene": "DIS",
    "Kerosene-Type Jet Fuel": "JET",
    "Residual Fuel Oil": "RES",
    "Propane": "LPG",
}
SUPPLY_PRODUCT = {
    "Finished Motor Gasoline": "GAS",
    "Gasoline Blending Components": "GAS",  # CBOB/RBOB; summed with finished gasoline
    "Total Gasoline": "GAS",
    "Distillate Fuel Oil": "DIS",
    "Kerosene-Type Jet Fuel": "JET",
    "Residual Fuel Oil": "RES",
    "Propane and Propylene": "LPG",
    "Propane": "LPG",  # used only where "Propane and Propylene" is not published
}
YIELD_PRODUCT = {
    "Finished Motor Gasoline": "GAS",
    "Distillate Fuel Oil": "DIS",
    "Kerosene-Type Jet Fuel": "JET",
    "Residual Fuel Oil": "RES",
    "Hydrocarbon Gas Liquids": "HGL",
}

# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

# EIA geography labels -> short area codes (PADDs and PADD 1 sub-districts)
AREA_LABELS = {
    "New England (PADD 1A)": "1A", "New England (PADD 1A )": "1A",
    "Central Atlantic (PADD 1B)": "1B", "Lower Atlantic (PADD 1C)": "1C",
    "East Coast (PADD 1)": "P1", "East Coast (PADD I)": "P1",
    "Midwest (PADD 2)": "P2", "Midwest (PADD II)": "P2",
    "Gulf Coast (PADD 3)": "P3", "Gulf Coast (PADD III)": "P3",
    "Rocky Mountain (PADD 4)": "P4", "Rocky Mountains (PADD 4)": "P4",
    "Rocky Mountain (PADD IV)": "P4",
    "West Coast (PADD 5)": "P5", "West Coast (PADD V)": "P5",
    "U.S.": "US", "U. S.": "US",
}

# Refining districts (EIA). P4 and P5 are each a single refining district.
DISTRICTS = {
    "EC": "East Coast",
    "AP": "Appalachian No. 1",
    "2A": "Indiana-Illinois-Kentucky",
    "2B": "Minnesota-Wisconsin-North/South Dakota",
    "2C": "Oklahoma-Kansas-Missouri",
    "3A": "Texas Inland",
    "3B": "Texas Gulf Coast",
    "3C": "Louisiana Gulf Coast",
    "3D": "North Louisiana-Arkansas",
    "3E": "New Mexico",
    "P4": "Rocky Mountain (PADD 4)",
    "P5": "West Coast (PADD 5)",
}
DISTRICT_PADD = {"EC": "P1", "AP": "P1", "2A": "P2", "2B": "P2", "2C": "P2",
                 "3A": "P3", "3B": "P3", "3C": "P3", "3D": "P3", "3E": "P3",
                 "P4": "P4", "P5": "P5"}
# Label variants used by EIA for the districts in different tables
_DISTRICT_LABELS = {
    # capacity / utilisation tables ("<label> Refining District ...")
    "East Coast Refining District": "EC", "East Coast": "EC",
    "Appalachian No. 1 Refining District": "AP", "Appalachian No. 1": "AP",
    "Indiana, Illinois, and Kentucky Refining District": "2A", "Indiana-Illinois-Kentucky": "2A",
    "Minnesota, Wisconsin, North and South Dakota Refining District": "2B",
    "Minnesota-Wisconsin-North Dakota-South Dakota": "2B",
    "Oklahoma, Kansas, Missouri Refining District": "2C", "Oklahoma-Kansas-Missouri": "2C",
    "Oklahoma, Kansas, and Missouri Refining District": "2C",
    "Texas Inland Refining District": "3A", "Texas Inland": "3A",
    "Texas Gulf Coast Refining District": "3B", "Texas Gulf Coast": "3B",
    "Louisiana Gulf Coast Refining District": "3C", "Louisiana Gulf Coast": "3C",
    "North Louisiana, Arkansas Refining District": "3D",
    "North Louisiana and Arkansas Refining District": "3D",
    "North Louisiana-Arkansas": "3D",
    "New Mexico Refining District": "3E", "New Mexico": "3E",
    "Rocky Mountains (PADD 4)": "P4", "Rocky Mountain (PADD 4)": "P4",
    "West Coast (PADD 5)": "P5",
}


def _strip_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def area_code(label: str) -> Optional[str]:
    return AREA_LABELS.get(_strip_label(label))


def district_code(label: str) -> Optional[str]:
    label = _strip_label(label)
    label = re.sub(r"^Refining District ", "", label)
    return _DISTRICT_LABELS.get(label)


# --------------------------------------------------------------------------- #
# Name patterns -> table rows
# --------------------------------------------------------------------------- #
_P = "|".join(re.escape(k) for k in SUPPLY_PRODUCT)
PATTERNS = {
    "prime": re.compile(
        r"^(?P<geo>.+?) (?P<prod>" + "|".join(re.escape(k) for k in PRIME_SUPPLIER_PRODUCT)
        + r") All Sales/Deliveries by Prime Supplier, Monthly$"),
    "capacity": re.compile(r"^(?P<geo>.+?) Operable Crude Oil Distillation Capacity, Monthly$"),
    "util": re.compile(r"^(?P<geo>.+?) Percent Utilization of Refinery Operable Capacity, Monthly$"),
    "crude_input": re.compile(
        r"^(?P<geo>.+?) Refinery and Blender Net Input of Crude Oil, Monthly$"),
    "production": re.compile(
        r"^(?P<geo>.+?) Refinery and Blender Net Production of (?P<prod>" + _P + r"), Monthly$"),
    "stocks": re.compile(r"^(?P<geo>.+?) Ending Stocks of (?P<prod>" + _P + r"), Monthly$"),
    "movement": re.compile(
        r"^(?P<to>.+?) Receipts by (?P<mode>Pipeline|Tanker and Barge) from (?P<frm>.+?) of (?P<prod>"
        + _P + r"), Monthly$"),
    "trade": re.compile(r"^(?P<geo>.+?) (?P<flow>Imports|Exports) of (?P<prod>" + _P + r"), Monthly$"),
    "fuel": re.compile(r"^(?P<geo>.+?\(PADD \w+\)) (?P<fuel>.+?) Consumed at Refineries, Annual$"),
    "yield": re.compile(r"^(?P<geo>.+?\(PADD \w+\)|U\.S\.) Refinery Yield of (?P<prod>"
                        + "|".join(re.escape(k) for k in YIELD_PRODUCT) + r"), Monthly$"),
    "util_weekly": re.compile(
        r"^(?P<geo>.+?) Percent Utilization of Refinery Operable Capacity, Weekly$"),
}
PRICE_SERIES = {
    "PET.RWTC.M": ("WTI_cushing", "usd_per_bbl"),
    "PET.RBRTE.M": ("Brent", "usd_per_bbl"),
    "PET.R0000____3.M": ("RAC_US", "usd_per_bbl"),
    "PET.R0010____3.M": ("RAC_P1", "usd_per_bbl"),
    "PET.R0020____3.M": ("RAC_P2", "usd_per_bbl"),
    "PET.R0030____3.M": ("RAC_P3", "usd_per_bbl"),
    "PET.R0040____3.M": ("RAC_P4", "usd_per_bbl"),
    "PET.R0050____3.M": ("RAC_P5", "usd_per_bbl"),
    "PET.EER_EPMRU_PF4_Y35NY_DPG.M": ("GAS_NYH", "usd_per_gal"),
    "PET.EER_EPMRU_PF4_RGC_DPG.M": ("GAS_USGC", "usd_per_gal"),
    "PET.EER_EPMRR_PF4_Y05LA_DPG.M": ("GAS_LA", "usd_per_gal"),
    "PET.EER_EPD2F_PF4_Y35NY_DPG.M": ("DIS_NYH", "usd_per_gal"),
    "PET.EER_EPD2DXL0_PF4_RGC_DPG.M": ("DIS_USGC", "usd_per_gal"),
    "PET.EER_EPJK_PF4_RGC_DPG.M": ("JET_USGC", "usd_per_gal"),
    "PET.EER_EPLLPA_PF4_Y44MB_DPG.M": ("LPG_MB", "usd_per_gal"),
}


# Single named monthly series (thousand barrels per month unless noted)
MISC_SERIES = {
    "PET.MFERIUS1.M": "US_ethanol_input_kbbl",
    "PET.M_EPLLNG_YIR_NUS_MBBL.M": "US_natural_gasoline_input_kbbl",
    "PET.MGFRPUS1.M": "US_finished_gasoline_prod_kbbl",
    "PET.M_EPLLPA_FPF_R10_MBBL.M": "P1_propane_field_prod_kbbl",
    "PET.M_EPLLPA_FPF_R20_MBBL.M": "P2_propane_field_prod_kbbl",
    "PET.M_EPLLPA_FPF_R30_MBBL.M": "P3_propane_field_prod_kbbl",
    "PET.M_EPLLPA_FPF_R40_MBBL.M": "P4_propane_field_prod_kbbl",
    "PET.M_EPLLPA_FPF_R50_MBBL.M": "P5_propane_field_prod_kbbl",
    "PET.MCRRIUS1.M": "US_crude_input_kbbl",
}


@dataclass
class BulkSelection:
    rows: Dict[str, List[dict]]
    used: List[dict]


def _iter_lines(path: str) -> Iterable[str]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
            with zf.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8"):
                    yield raw
    else:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                yield raw


def _period_to_ts(p: str, freq: str) -> pd.Timestamp:
    if freq == "M":
        return pd.Timestamp(year=int(p[:4]), month=int(p[4:6]), day=1)
    if freq == "W":
        return pd.Timestamp(p)
    if freq == "A":
        return pd.Timestamp(year=int(p[:4]), month=1, day=1)
    raise ValueError(freq)


def scan_bulk(path: str) -> BulkSelection:
    """Single pass over PET.txt/PET.zip collecting only the series OptiGreen needs."""
    rows: Dict[str, List[dict]] = {k: [] for k in list(PATTERNS) + ["price", "misc"]}
    used: List[dict] = []
    for line in _iter_lines(path):
        if not line.startswith('{"series_id"'):
            continue
        d = json.loads(line)
        sid, name, units, f = d["series_id"], d.get("name", ""), d.get("units", ""), d.get("f")
        data = d.get("data") or []
        meta = None
        if sid in PRICE_SERIES and f == "M":
            key, unit = PRICE_SERIES[sid]
            meta = ("price", {"series": key, "unit": unit})
        elif sid in MISC_SERIES:
            meta = ("misc", {"series": MISC_SERIES[sid]})
        elif f == "M":
            for table in ("prime", "capacity", "util", "crude_input", "production",
                          "stocks", "movement", "trade", "yield"):
                m = PATTERNS[table].match(name)
                if not m:
                    continue
                gd = m.groupdict()
                if table == "prime" and units != "Thousand Gallons per Day":
                    break
                if table in ("production", "stocks", "movement", "trade") and units != "Thousand Barrels":
                    break
                if table == "crude_input" and units != "Thousand Barrels per Day":
                    break
                if table == "yield" and units != "Percent":
                    break
                meta = (table, gd)
                break
        elif f == "A":
            m = PATTERNS["fuel"].match(name)
            if m:
                meta = ("fuel", m.groupdict() | {"units": units})
        elif f == "W":
            m = PATTERNS["util_weekly"].match(name)
            if m:
                meta = ("util_weekly", m.groupdict())
        if meta is None:
            continue
        table, info = meta
        used.append({"table": table, "series_id": sid, "name": name, "units": units})
        for p, v in data:
            if v is None:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue  # 'W' (withheld) / 'NA'
            rows[table].append({**info, "period": _period_to_ts(p, f), "value": val, "series_id": sid})
    return BulkSelection(rows=rows, used=used)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _days(month: pd.Series) -> np.ndarray:
    return month.dt.days_in_month.to_numpy()


def kbbl_to_kt(kbbl, product: str):
    return np.asarray(kbbl, dtype=float) * M3_PER_BBL * DENSITY_T_PER_M3[product]


def kt_to_kbbl(kt, product: str):
    return np.asarray(kt, dtype=float) / (M3_PER_BBL * DENSITY_T_PER_M3[product])


def build_tables(sel: BulkSelection) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    # ---- state demand (prime supplier sales) ----
    df = pd.DataFrame(sel.rows["prime"])
    df["state"] = df["geo"].map(STATES)
    df = df.dropna(subset=["state"])
    df["product"] = df["prod"].map(PRIME_SUPPLIER_PRODUCT)
    df = df.rename(columns={"period": "month", "value": "kgal_per_day"})
    df = df[["month", "state", "product", "kgal_per_day"]].drop_duplicates(["month", "state", "product"])
    m3 = df["kgal_per_day"].to_numpy() * _days(df["month"]) * M3_PER_KGAL
    df["kt"] = m3 * df["product"].map(DENSITY_T_PER_M3).to_numpy() / 1000.0
    out["demand_state_monthly"] = df.sort_values(["state", "product", "month"]).reset_index(drop=True)
    # PADD / sub-PADD / U.S. totals. These include state values that EIA withholds
    # for confidentiality, so "area total - sum of published states" is the
    # withheld remainder (used as a regional pool in the optimisation model).
    ar = pd.DataFrame(sel.rows["prime"])
    ar["area"] = ar["geo"].map(area_code)
    ar = ar.dropna(subset=["area"]).copy()
    ar["product"] = ar["prod"].map(PRIME_SUPPLIER_PRODUCT)
    ar = (ar.rename(columns={"period": "month", "value": "kgal_per_day"})
          [["month", "area", "product", "kgal_per_day"]]
          .drop_duplicates(["month", "area", "product"]))
    m3 = ar["kgal_per_day"].to_numpy() * _days(ar["month"]) * M3_PER_KGAL
    ar["kt"] = m3 * ar["product"].map(DENSITY_T_PER_M3).to_numpy() / 1000.0
    out["demand_area_monthly"] = ar.sort_values(["area", "product", "month"]).reset_index(drop=True)

    # ---- districts ----
    def _district_frame(table: str, value_name: str) -> pd.DataFrame:
        d = pd.DataFrame(sel.rows[table])
        d["district"] = d["geo"].map(district_code)
        d = d.dropna(subset=["district"])
        return (d.rename(columns={"period": "month", "value": value_name})
                [["month", "district", value_name]].drop_duplicates(["month", "district"]))

    cap = _district_frame("capacity", "capacity_kbcd")
    util = _district_frame("util", "utilization_pct")
    crude = _district_frame("crude_input", "crude_input_kbd")
    prod = pd.DataFrame(sel.rows["production"])
    prod["district"] = prod["geo"].map(district_code)
    prod = prod.dropna(subset=["district"])
    prod["product"] = prod["prod"].map(SUPPLY_PRODUCT)
    # keep finished gasoline only for production (blending components are inputs)
    prod = prod[prod["prod"] != "Gasoline Blending Components"]
    prod = prod[~prod["prod"].isin(["Total Gasoline", "Propane"])]
    prod = (prod.groupby(["period", "district", "product"])["value"].sum().unstack("product")
            .add_prefix("prod_").add_suffix("_kbbl").reset_index().rename(columns={"period": "month"}))
    dist = cap.merge(util, on=["month", "district"], how="outer") \
              .merge(crude, on=["month", "district"], how="outer") \
              .merge(prod, on=["month", "district"], how="outer")
    out["district_monthly"] = dist.sort_values(["district", "month"]).reset_index(drop=True)

    # ---- stocks ----
    st = pd.DataFrame(sel.rows["stocks"])
    st["area"] = st["geo"].map(area_code)
    st = st.dropna(subset=["area"])
    st["product"] = st["prod"].map(SUPPLY_PRODUCT)
    # gasoline: use Total Gasoline stocks (finished + blending components)
    st = st[~((st["product"] == "GAS") & (st["prod"] != "Total Gasoline"))]
    st = st[st["prod"] != "Propane"]
    st = (st.rename(columns={"period": "month", "value": "kbbl"})
          [["month", "area", "product", "kbbl"]].drop_duplicates(["month", "area", "product"]))
    out["stocks_monthly"] = st.sort_values(["area", "product", "month"]).reset_index(drop=True)

    # ---- inter-PADD movements ----
    mv = pd.DataFrame(sel.rows["movement"])
    mv["to_area"] = mv["to"].map(area_code)
    mv["from_area"] = mv["frm"].map(area_code)
    mv = mv.dropna(subset=["to_area", "from_area"])
    mv["product"] = mv["prod"].map(SUPPLY_PRODUCT)
    mv = mv[~mv["prod"].isin(["Total Gasoline", "Propane"])]
    mv["mode"] = mv["mode"].map({"Pipeline": "pipeline", "Tanker and Barge": "marine"})
    mv = (mv.groupby(["period", "from_area", "to_area", "mode", "product"])["value"].sum()
          .reset_index().rename(columns={"period": "month", "value": "kbbl"}))
    out["movements_monthly"] = mv.sort_values(["from_area", "to_area", "mode", "product", "month"])

    # ---- imports / exports ----
    tr = pd.DataFrame(sel.rows["trade"])
    tr["padd"] = tr["geo"].map(area_code)
    tr = tr.dropna(subset=["padd"])
    tr = tr[tr["prod"] != "Total Gasoline"]
    # LPG: prefer "Propane and Propylene"; fall back to "Propane" where the former is absent
    has_pp = set(tr.loc[tr["prod"] == "Propane and Propylene", ["geo", "flow"]].itertuples(index=False, name=None))
    drop = (tr["prod"] == "Propane") & tr[["geo", "flow"]].apply(tuple, axis=1).isin(has_pp)
    tr = tr[~drop]
    tr["product"] = tr["prod"].map(SUPPLY_PRODUCT)
    tr["flow"] = tr["flow"].str.lower()
    tr = (tr.groupby(["period", "padd", "flow", "product"])["value"].sum().reset_index()
          .rename(columns={"period": "month", "value": "kbbl"}))
    out["trade_monthly"] = tr.sort_values(["padd", "flow", "product", "month"])

    # ---- prices ----
    pr = pd.DataFrame(sel.rows["price"]).rename(columns={"period": "month", "value": "price"})
    out["prices_monthly"] = pr[["month", "series", "unit", "price"]].sort_values(["series", "month"])

    # ---- misc named series ----
    mi = pd.DataFrame(sel.rows["misc"]).rename(columns={"period": "month", "value": "value"})
    out["misc_monthly"] = mi[["month", "series", "value"]].sort_values(["series", "month"])

    # ---- refinery fuel use (annual) ----
    fu = pd.DataFrame(sel.rows["fuel"])
    fu["padd"] = fu["geo"].map(area_code)
    fu = fu.dropna(subset=["padd"])
    fu["year"] = fu["period"].dt.year
    out["refinery_fuel_annual"] = (fu.rename(columns={"value": "value"})
                                   [["year", "padd", "fuel", "units", "value"]]
                                   .sort_values(["padd", "fuel", "year"]))

    # ---- PADD refinery yields (% of crude + unfinished-oil input) ----
    yl = pd.DataFrame(sel.rows["yield"])
    yl["padd"] = yl["geo"].map(area_code)
    yl["product"] = yl["prod"].map(YIELD_PRODUCT)
    out["yields_padd_monthly"] = (yl.dropna(subset=["padd"])
                                  .rename(columns={"period": "month", "value": "yield_pct"})
                                  [["month", "padd", "product", "yield_pct"]]
                                  .drop_duplicates(["month", "padd", "product"])
                                  .sort_values(["padd", "product", "month"]))

    # ---- weekly PADD utilisation ----
    wk = pd.DataFrame(sel.rows["util_weekly"])
    wk["padd"] = wk["geo"].map(area_code)
    out["utilization_weekly"] = (wk.dropna(subset=["padd"])
                                 .rename(columns={"period": "week", "value": "utilization_pct"})
                                 [["week", "padd", "utilization_pct"]].sort_values(["padd", "week"]))

    out["series_used"] = pd.DataFrame(sel.used).sort_values(["table", "series_id"])
    return out


def build_from_bulk(bulk_path: str, out_dir: str) -> Dict[str, pd.DataFrame]:
    """Read PET.zip / PET.txt and write the tidy tables to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    tables = build_tables(scan_bulk(bulk_path))
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv.gz"), index=False)
    return tables


def load_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load the tidy tables written by `build_from_bulk`."""
    date_cols = {"month", "week"}
    out = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".csv.gz"):
            key = fn[:-7]
        elif fn.endswith(".csv"):
            key = fn[:-4]
        else:
            continue
        df = pd.read_csv(os.path.join(data_dir, fn))
        for c in date_cols & set(df.columns):
            df[c] = pd.to_datetime(df[c])
        out[key] = df
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Extract OptiGreen-Chem tables from the EIA PET bulk file")
    ap.add_argument("bulk", help="path to PET.zip or PET.txt")
    ap.add_argument("--out", default="data/real/eia")
    a = ap.parse_args()
    t = build_from_bulk(a.bulk, a.out)
    for k, v in t.items():
        print(f"{k:28s} {len(v):>8d} rows")
