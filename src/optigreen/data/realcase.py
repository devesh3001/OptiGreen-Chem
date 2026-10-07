"""
Model-ready panels for the OptiGreen-Chem real-data case, built from the tidy
EIA tables written by `optigreen.data.eia`.

Every function here only reshapes, converts units or derives quantities from
published EIA data; the few modelling choices (imputation of withheld values,
splitting of shared stock areas, outage definition) are explicit parameters and
are documented in the docstrings so they can be reported.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from optigreen.data.eia import (CRUDE_DENSITY_T_PER_M3, DENSITY_T_PER_M3, DISTRICT_PADD,
                                M3_PER_BBL, PRODUCTS)
from optigreen.data.network import DEMAND_AREAS, HUBS, STATE_INFO, primary_hub_for_state

DEMAND_START = pd.Timestamp("1993-01-01")
DEMAND_END = pd.Timestamp("2022-03-01")  # EIA ended Form EIA-782C (prime supplier) here


def kt_per_kbbl(product: str) -> float:
    return M3_PER_BBL * DENSITY_T_PER_M3[product]


# --------------------------------------------------------------------------- #
# Demand
# --------------------------------------------------------------------------- #
def _impute_series(s: pd.Series) -> (pd.Series, pd.Series):
    """Fill withheld months: log-linear interpolation for gaps <= 3 months,
    otherwise the value 12 months earlier (or later), else nearest value."""
    s = s.astype(float).copy()
    orig_na = s.isna()
    pos = s.clip(lower=1e-6)
    filled = np.exp(np.log(pos).interpolate(limit=3, limit_area="inside"))
    filled[pos.isna() & ~filled.notna()] = np.nan
    for lag in (12, -12, 24, -24):
        filled = filled.fillna(filled.shift(lag))
    filled = filled.ffill().bfill()
    # keep true zeros as zeros
    filled[(~orig_na) & (s <= 0)] = 0.0
    return filled, orig_na


def demand_panel(tables: Dict[str, pd.DataFrame], max_missing: float = 0.15,
                 select_end: str = "2015-12-01") -> pd.DataFrame:
    """Monthly demand (kt) for every (region, product), 1993-01 .. 2022-03.

    Regions are states whose series has at most `max_missing` withheld months
    in 1993..`select_end`, plus one pool per EIA demand area that carries the
    remainder (area total minus included states, floored at zero). Residual
    fuel oil is withheld for most states, so it is modelled almost entirely
    through the area pools.
    """
    st = tables["demand_state_monthly"].copy()
    ar = tables["demand_area_monthly"].copy()
    months = pd.date_range(DEMAND_START, DEMAND_END, freq="MS")
    st = st[(st["month"] >= DEMAND_START) & (st["month"] <= DEMAND_END)]
    ar = ar[(ar["month"] >= DEMAND_START) & (ar["month"] <= DEMAND_END)]

    rows = []
    included = {}
    sel_mask = months <= pd.Timestamp(select_end)
    for (state, prod), g in st.groupby(["state", "product"]):
        s = g.set_index("month")["kt"].reindex(months)
        miss = s[sel_mask].isna().mean()
        if miss > max_missing or prod == "RES":
            continue
        filled, na = _impute_series(s)
        included[(state, prod)] = filled
        rows.append(pd.DataFrame({"month": months, "region": state, "product": prod,
                                  "kt": filled.to_numpy(), "imputed": na.to_numpy(),
                                  "area": STATE_INFO[state][0], "kind": "state"}))
    for area in DEMAND_AREAS:
        for prod in PRODUCTS:
            g = ar[(ar["area"] == area) & (ar["product"] == prod)]
            tot, na = _impute_series(g.set_index("month")["kt"].reindex(months))
            inc = [v for (s_, p_), v in included.items() if p_ == prod and STATE_INFO[s_][0] == area]
            rest = tot - (sum(inc) if inc else 0.0)
            rest = rest.clip(lower=0.0)
            rows.append(pd.DataFrame({"month": months, "region": f"POOL_{area}", "product": prod,
                                      "kt": rest.to_numpy(), "imputed": na.to_numpy(),
                                      "area": area, "kind": "pool"}))
    out = pd.concat(rows, ignore_index=True)
    out["hub"] = out.apply(lambda r: primary_hub_for_state(r["region"]) if r["kind"] == "state"
                           else _pool_hub(r["area"]), axis=1)
    return out.sort_values(["region", "product", "month"]).reset_index(drop=True)


def _pool_hub(area: str) -> str:
    return {"1A": "H1A", "1B": "H1B", "1C": "H1C", "P2": "H2N", "P3": "H3",
            "P4": "H4", "P5": "H5"}[area]


# --------------------------------------------------------------------------- #
# Refining districts
# --------------------------------------------------------------------------- #
def district_panel(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = tables["district_monthly"].copy()
    d["days"] = d["month"].dt.days_in_month
    d["capacity_kbbl"] = d["capacity_kbcd"] * d["days"]
    d["crude_kbbl"] = d["crude_input_kbd"] * d["days"]
    d["padd"] = d["district"].map(DISTRICT_PADD)
    return d.sort_values(["district", "month"]).reset_index(drop=True)


def us_utilization(tables: Dict[str, pd.DataFrame]) -> pd.Series:
    d = district_panel(tables)
    g = d.dropna(subset=["capacity_kbcd", "utilization_pct"])
    w = g.assign(x=g["capacity_kbcd"] * g["utilization_pct"]).groupby("month")
    return (w["x"].sum() / w["capacity_kbcd"].sum()).rename("us_util")


def outage_labels(tables: Dict[str, pd.DataFrame], threshold_pp: float = 10.0,
                  window: int = 12) -> pd.DataFrame:
    """District-specific refinery outages from monthly utilisation.

    deviation_d(t)   = util_d(t) - median(util_d, t-12..t-1)
    deviation_US(t)  = same for the capacity-weighted U.S. utilisation
    idiosyncratic(t) = deviation_d(t) - deviation_US(t)
    outage(t)        = idiosyncratic(t) <= -threshold_pp

    Subtracting the national deviation removes demand-driven run cuts (e.g.
    April 2020) so the label captures events that hit one district harder than
    the rest of the system: hurricanes, freezes, fires, turnarounds, closures.
    severity(t) = fraction of the district's normal throughput lost.
    """
    d = district_panel(tables)[["month", "district", "padd", "capacity_kbcd", "utilization_pct"]].copy()
    us = us_utilization(tables)
    us_ref = us.shift(1).rolling(window, min_periods=window).median()
    us_dev = (us - us_ref).rename("us_dev")
    out = []
    for dist, g in d.groupby("district"):
        g = g.sort_values("month").set_index("month")
        ref = g["utilization_pct"].shift(1).rolling(window, min_periods=window).median()
        g["ref_util"] = ref
        g["dev"] = g["utilization_pct"] - ref
        g = g.join(us_dev, how="left")
        g["idio_dev"] = g["dev"] - g["us_dev"]
        g["outage"] = (g["idio_dev"] <= -threshold_pp).astype(float)
        g.loc[g["idio_dev"].isna(), "outage"] = np.nan
        g["severity"] = ((g["ref_util"] - g["utilization_pct"]) / g["ref_util"]).clip(0, 1)
        g.loc[g["outage"] != 1, "severity"] = 0.0
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def misc_series(tables: Dict[str, pd.DataFrame], name: str) -> pd.Series:
    m = tables["misc_monthly"]
    return m[m["series"] == name].set_index("month")["value"].sort_index()


def gasoline_blend_uplift(tables: Dict[str, pd.DataFrame]) -> pd.Series:
    """Finished-gasoline volume per volume of refinery gasoline from crude.

    EIA's refinery yield of finished motor gasoline nets out ethanol and NGL
    blendstocks, which are added at refineries and terminals. The uplift is
        1 + (U.S. fuel-ethanol input + U.S. natural-gasoline input)
            / sum_d(PADD gasoline yield x district crude input)
    computed monthly from EIA data.
    """
    d = district_panel(tables)
    y = yields_panel(tables)
    yg = y[y["product"] == "GAS"].rename(columns={"yield_pct": "y_gas"})[["month", "padd", "y_gas"]]
    d = d.merge(yg, on=["month", "padd"], how="left").dropna(subset=["y_gas", "crude_kbbl"])
    base = d.assign(g=d["y_gas"] / 100 * d["crude_kbbl"]).groupby("month")["g"].sum()
    eth = misc_series(tables, "US_ethanol_input_kbbl").reindex(base.index).fillna(0.0)
    ngl = misc_series(tables, "US_natural_gasoline_input_kbbl").reindex(base.index).fillna(0.0)
    return (1.0 + (eth + ngl) / base).rename("gas_uplift")


def ngl_propane_supply(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Field (gas-plant) production of propane by PADD, kt per month. Most U.S.
    propane comes from gas processing, not refineries, so it is a supply source
    at the PADD's main hub in the optimisation model."""
    rows = []
    for padd in ["P1", "P2", "P3", "P4", "P5"]:
        s = misc_series(tables, f"{padd}_propane_field_prod_kbbl")
        rows.append(pd.DataFrame({"month": s.index, "padd": padd, "kt": s.to_numpy() * kt_per_kbbl("LPG")}))
    return pd.concat(rows, ignore_index=True)


def yields_panel(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return tables["yields_padd_monthly"].copy()


def district_yields(tables: Dict[str, pd.DataFrame], origin: pd.Timestamp,
                    months: int = 12) -> pd.DataFrame:
    """Product yield (kt of product per kbbl of crude run) for each district,
    averaged over the `months` months before `origin`.

    DIS, JET, RES, LPG: district net production / district crude input.
    GAS: PADD refinery yield of finished gasoline x blend uplift (ethanol/NGL).
    """
    d = district_panel(tables)
    lo, hi = origin - pd.DateOffset(months=months), origin - pd.DateOffset(months=1)
    w = d[(d["month"] >= lo) & (d["month"] <= hi)]
    y = yields_panel(tables)
    yw = y[(y["month"] >= lo) & (y["month"] <= hi) & (y["product"] == "GAS")]
    ygas = yw.groupby("padd")["yield_pct"].mean() / 100.0
    up = gasoline_blend_uplift(tables)
    uplift = float(up[(up.index >= lo) & (up.index <= hi)].mean())
    rows = []
    for dist, g in w.groupby("district"):
        crude = g["crude_kbbl"].sum()
        r = {"district": dist}
        for p in ["DIS", "JET", "RES", "LPG"]:
            col = f"prod_{p}_kbbl"
            vol = g[col].fillna(0).clip(lower=0).sum()
            r[p] = (vol / crude if crude > 0 else 0.0) * kt_per_kbbl(p)
        r["GAS"] = float(ygas.get(DISTRICT_PADD[dist], np.nan)) * uplift * kt_per_kbbl("GAS")
        rows.append(r)
    out = pd.DataFrame(rows).set_index("district")
    out.attrs["gas_uplift"] = uplift
    return out[PRODUCTS]


# --------------------------------------------------------------------------- #
# Stocks (hubs)
# --------------------------------------------------------------------------- #
def hub_stock_panel(tables: Dict[str, pd.DataFrame], demand: pd.DataFrame) -> pd.DataFrame:
    """Month-end stocks (kt) per hub and product.

    Areas 1A/1B/P3/P4 map to one hub. Split areas (1C, P2, P5) and the jet-fuel
    stocks of PADD 1 (published only at PADD level) are divided by each hub's
    share of the area's demand over the previous 12 months.
    """
    st = tables["stocks_monthly"].copy()
    st["kt"] = [kb * kt_per_kbbl(p) for kb, p in zip(st["kbbl"], st["product"])]
    dem = demand.copy()
    dem["hub_area"] = dem["hub"].map(lambda h: HUBS[h][0])
    # demand share of each hub within its stock area (trailing 12 months)
    hd = dem.groupby(["month", "hub", "hub_area", "product"])["kt"].sum().reset_index()
    hd = hd.sort_values("month")
    hd["kt12"] = hd.groupby(["hub", "product"])["kt"].transform(lambda x: x.rolling(12, min_periods=1).mean())
    hd["share"] = hd["kt12"] / hd.groupby(["month", "hub_area", "product"])["kt12"].transform("sum")
    # PADD1 sub-area share (for jet fuel)
    p1 = hd[hd["hub_area"].isin(["1A", "1B", "1C"])].copy()
    p1["p1_share"] = p1["kt12"] / p1.groupby(["month", "product"])["kt12"].transform("sum")

    rows = []
    for h, (area, *_rest) in HUBS.items():
        for p in PRODUCTS:
            if p == "JET" and area in ("1A", "1B", "1C"):
                base = st[(st["area"] == "P1") & (st["product"] == p)][["month", "kt"]]
                sh = p1[(p1["hub"] == h) & (p1["product"] == p)][["month", "p1_share"]].rename(
                    columns={"p1_share": "share"})
            else:
                base = st[(st["area"] == area) & (st["product"] == p)][["month", "kt"]]
                sh = hd[(hd["hub"] == h) & (hd["product"] == p)][["month", "share"]]
            m = base.merge(sh, on="month", how="left")
            n_hubs = sum(1 for v in HUBS.values() if v[0] == area)
            m["share"] = m["share"].fillna(1.0 / n_hubs if not (p == "JET" and area.startswith("1")) else 1 / 4)
            m["kt"] = m["kt"] * m["share"]
            m["hub"], m["product"] = h, p
            rows.append(m[["month", "hub", "product", "kt"]])
    return pd.concat(rows, ignore_index=True).sort_values(["hub", "product", "month"])


# --------------------------------------------------------------------------- #
# Inter-PADD movements, trade, prices
# --------------------------------------------------------------------------- #
def movements_panel(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    mv = tables["movements_monthly"].copy()
    mv["kt"] = [kb * kt_per_kbbl(p) for kb, p in zip(mv["kbbl"], mv["product"])]
    # PADD-1 marine receipts are published both for PADD 1 and its sub-areas; keep sub-areas
    mv = mv[~((mv["to_area"] == "P1") & (mv["mode"] == "marine"))]
    return mv


def trade_panel(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    tr = tables["trade_monthly"].copy()
    tr = tr[tr["padd"] != "US"]
    tr["kt"] = [kb * kt_per_kbbl(p) for kb, p in zip(tr["kbbl"], tr["product"])]
    return tr


PRICE_SERIES_FOR = {  # product, region -> preferred spot series (fallbacks follow)
    ("GAS", "east"): ["GAS_NYH", "GAS_USGC"], ("GAS", "gulf"): ["GAS_USGC", "GAS_NYH"],
    ("GAS", "west"): ["GAS_LA", "GAS_USGC"],
    ("DIS", "east"): ["DIS_NYH", "DIS_USGC"], ("DIS", "gulf"): ["DIS_USGC", "DIS_NYH"],
    ("DIS", "west"): ["DIS_USGC", "DIS_NYH"],
    ("JET", "east"): ["JET_USGC"], ("JET", "gulf"): ["JET_USGC"], ("JET", "west"): ["JET_USGC"],
    ("LPG", "east"): ["LPG_MB"], ("LPG", "gulf"): ["LPG_MB"], ("LPG", "west"): ["LPG_MB"],
}


def price_table(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Monthly prices in USD per tonne for products and USD per bbl for crude.

    Residual fuel oil has no spot series in the bulk file; it is priced at
    0.85 x Brent per barrel-equivalent (long-run U.S. Gulf HSFO/Brent ratio,
    stated assumption).
    """
    pr = tables["prices_monthly"].pivot_table(index="month", columns="series", values="price")
    out = pd.DataFrame(index=pr.index)
    for (p, reg), cands in PRICE_SERIES_FOR.items():
        s = None
        for c in cands:
            if c in pr:
                s = pr[c] if s is None else s.fillna(pr[c])
        out[f"{p}_{reg}"] = s * 42.0 / kt_per_kbbl(p)  # $/gal -> $/bbl -> $/t  (t/bbl == kt/kbbl)
    brent = pr["Brent"].fillna(pr["WTI_cushing"])
    for reg in ("east", "gulf", "west"):
        out[f"RES_{reg}"] = 0.85 * brent / kt_per_kbbl("RES")
    out["crude_US"] = pr["RAC_US"]
    for padd in ["P1", "P2", "P3", "P4", "P5"]:
        col = f"RAC_{padd}"
        ratio = (pr[col] / pr["RAC_US"]).dropna()
        fill = pr["RAC_US"] * (ratio.iloc[:36].mean() if len(ratio) else 1.0)
        out[f"crude_{padd}"] = pr[col].fillna(fill) if col in pr else fill
    return out.sort_index()


def hub_price_region(hub: str) -> str:
    return {"H1A": "east", "H1B": "east", "H1C": "east", "H1F": "east", "H2N": "gulf",
            "H2S": "gulf", "H3": "gulf", "H4": "gulf", "H5": "west", "H5N": "west"}[hub]


# --------------------------------------------------------------------------- #
# Refinery CO2 intensity (energy balance -> emissions)
# --------------------------------------------------------------------------- #
# Heat contents (EIA MER Table A1 / EPA Hub) and CO2 factors (EPA GHG Emission
# Factors Hub 2024 Table 1, Table 6 and Table 7; 40 CFR 98 Table C-1 for still gas).
FUEL_FACTORS = {
    # fuel label in EIA          (MMBtu per unit, kg CO2 per MMBtu)
    "Natural Gas": (1026.0, 53.06),            # per million cubic feet
    "Still Gas": (6000.0, 66.72),              # per thousand bbl (FOE 6.0 MMBtu/bbl)
    "Petroleum Coke": (6024.0, 102.41),        # per thousand bbl (marketable + catalyst)
    "Distillate": (5825.0, 73.96),
    "Residual Fuel Oil": (6287.0, 75.10),
    "HGL's": (3836.0, 61.71),
    "Other Products": (5800.0, 74.00),
    "Crude Oil": (5800.0, 74.54),
    "Coal": (24930.0, 93.28),                  # per thousand short tons
    "Purchased Steam": (1194.0, 66.33),        # per million lb (~1,194 Btu/lb)
}
GRID_T_CO2_PER_MWH = 823.1 * 0.45359237 / 1000.0  # EPA Hub 2024 Table 6 U.S. average


def refinery_co2_intensity(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """t CO2 per thousand barrels of crude run, by PADD and year."""
    fu = tables["refinery_fuel_annual"].copy()
    fu = fu[fu["padd"].isin(["P1", "P2", "P3", "P4", "P5", "US"])]
    rows = []
    for (padd, year), g in fu.groupby(["padd", "year"]):
        vals = dict(zip(g["fuel"], g["value"]))
        t = 0.0
        for fuel, (mmbtu, kg) in FUEL_FACTORS.items():
            if fuel == "Petroleum Coke":
                q = vals.get("Petroleum Coke", np.nan)
                if np.isnan(q):
                    q = vals.get("Marketable Petroleum Coke", 0.0) + vals.get("Catalyst Petroleum Coke", 0.0)
            else:
                q = vals.get(fuel, 0.0)
            q = 0.0 if q is None or np.isnan(q) else q
            t += q * mmbtu * kg / 1000.0
        t += vals.get("Purchased Electricity", 0.0) * GRID_T_CO2_PER_MWH * 1000.0  # MkWh = GWh
        rows.append({"padd": padd, "year": int(year), "t_co2": t})
    em = pd.DataFrame(rows)
    em = em[em["padd"] != "US"]
    us_em = em.groupby("year")["t_co2"].sum().reset_index().assign(padd="US")
    em = pd.concat([em, us_em], ignore_index=True)
    d = district_panel(tables)
    d["year"] = d["month"].dt.year
    crude = d.groupby(["padd", "year"])["crude_kbbl"].sum().reset_index()
    us = crude.groupby("year")["crude_kbbl"].sum().reset_index().assign(padd="US")
    crude = pd.concat([crude, us])
    em = em.merge(crude, on=["padd", "year"], how="left")
    em["t_co2_per_kbbl"] = em["t_co2"] / em["crude_kbbl"]
    return em.sort_values(["padd", "year"])


def crude_kt_per_kbbl() -> float:
    return M3_PER_BBL * CRUDE_DENSITY_T_PER_M3
