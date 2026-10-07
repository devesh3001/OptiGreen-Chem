"""
OptiGreen-Chem planning model for the real-data case (U.S. refined products).

Tactical plan, monthly buckets, horizon T (default 3 months), made at origin o
with information up to o-1.

Decision variables (units)
--------------------------
x[d,t]        crude run of refining district d (kbbl)                continuous
f[a,k,t]      district -> hub flow on arc a of product k (kt)          continuous
g[a,k,t]      district -> region direct rack sales (kt)                continuous
q[a,k,t]      hub -> region deliveries (kt)                            continuous
inv[h,k,t]    hub month-end inventory (kt)                             continuous
imp[h,k,t]    product imports at hub h (kt)                            continuous
ngl[h,t]      gas-plant propane bought at hub h (kt)                   continuous
xs[h,k,t]     surplus exported from hub h (kt)                         continuous
es[h,k,t]     shortfall against export commitments (kt)                continuous
u[r,k,t]      unmet demand of region r (kt)                            continuous
n[l,t]        marine cargoes on lane l (tanker / barge tow)            INTEGER
m[h,t]        import cargoes at port hub h                             INTEGER

Constraints
-----------
cap_eff*minrun <= x <= cap_eff         cap_eff = capacity x u_max [x (1 - p*sev) if risk-aware]
production     prod[d,k,t] = yield[d,k] x x[d,t]             (joint products, real yields)
district bal.  prod = sum_out f + sum_out g
hub balance    inv[t] = inv[t-1] + in f + imp + ngl - out q - exports - xs
inventory      floor <= inv <= cap ; inv[T] >= seasonal target
demand         sum_in q + sum_in g + u = D[r,k,t]
exports        exported + es = commitment
lanes          sum over arcs/products of flow (kbbl) <= historical lane capacity
marine         flow on lane (kbbl) <= cargo size x n[l,t]
imports        sum_k imp (kbbl) <= cargo size x m[h,t] ; imp <= import cap

Objective (thousand USD)
------------------------
crude + refining opex + pipeline/truck transport + voyage lump sums + imports
+ propane purchases + holding + shortage penalty + export-shortfall penalty
+ lambda_CO2 x CO2 (kt)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from optigreen.data.eia import PRODUCTS
from optigreen.data import network as net
from optigreen.data import realcase as rc

KT_PER_KBBL = {k: rc.kt_per_kbbl(k) for k in PRODUCTS}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class PlanningInputs:
    origin: pd.Timestamp
    months: List[pd.Timestamp]
    districts: pd.DataFrame          # index district: padd, cap_kbbl_t (list), umax, crude_usd_bbl, co2_t_per_kbbl
    yields: pd.DataFrame             # district x product (kt per kbbl crude)
    hubs: pd.DataFrame               # index (hub, product): inv0, floor, cap, target
    demand: pd.DataFrame             # region, product, month, kt, area, hub
    arcs_dh: pd.DataFrame            # id, d, h, mode, km, lane, usd_per_t, co2_t_per_t
    arcs_dr: pd.DataFrame            # id, d, r, km, usd_per_t, co2_t_per_t
    arcs_hr: pd.DataFrame            # id, h, r, km, usd_per_t, co2_t_per_t
    lanes: pd.DataFrame              # lane: cap_kbbl, marine, cargo_kbbl, voyage_kusd, mode
    imports: pd.DataFrame            # hub, product: cap_kt, usd_per_t, co2_t_per_t, marine
    exports: pd.DataFrame            # hub, product, month: commit_kt, spot_usd_per_t
    ngl: pd.DataFrame                # hub, month: cap_kt, usd_per_t, co2_t_per_t
    spot: Dict[Tuple[str, str], float]  # (price region, product) -> $/t
    risk: Optional[pd.DataFrame] = None  # district, month: p, sev
    params: Dict = field(default_factory=dict)
    export_spare: Optional[pd.DataFrame] = None  # hub, product: cap_kt, netback_usd_per_t

    @property
    def T(self) -> List[int]:
        return list(range(len(self.months)))


def _area_price_region(area: str) -> str:
    return {"1A": "east", "1B": "east", "1C": "east", "P2": "gulf", "P3": "gulf", "P4": "gulf",
            "P5": "west"}[area]


def build_inputs(tables: Dict[str, pd.DataFrame], demand_panel: pd.DataFrame,
                 forecast: Optional[pd.DataFrame], origin: pd.Timestamp, horizon: int = 3,
                 demand_col: str = "P50", risk: Optional[pd.DataFrame] = None,
                 labels: Optional[pd.DataFrame] = None, minrun: float = 0.7,
                 lane_lookback: int = 36) -> PlanningInputs:
    """Assemble every model parameter from EIA data available before `origin`.

    forecast: rows (region, product_name, target_month, P10/P50/P90). If None,
    the actual demand is used (perfect-information benchmark).
    risk: rows (district, target_month, p) from a risk model.
    """
    o = pd.Timestamp(origin)
    months = [o + pd.DateOffset(months=i) for i in range(horizon)]
    prev = o - pd.DateOffset(months=1)
    C = net.COST_ASSUMPTIONS
    prices = rc.price_table(tables)
    pr_prev = prices.loc[:prev].iloc[-1]
    spot = {(reg, k): float(pr_prev[f"{k}_{reg}"]) for reg in ("east", "gulf", "west") for k in PRODUCTS}

    # ---------------- districts ----------------
    dp = rc.district_panel(tables)
    co2 = rc.refinery_co2_intensity(tables)
    yl = rc.district_yields(tables, o)
    rows = []
    for d, info in net.DISTRICT_INFO.items():
        g = dp[(dp["district"] == d) & (dp["month"] <= prev)]
        cap_kbcd = float(g["capacity_kbcd"].dropna().iloc[-1])
        util5 = g[g["month"] > prev - pd.DateOffset(months=60)]["utilization_pct"].dropna()
        umax = float(min(0.99, np.percentile(util5, 95) / 100.0))
        padd = info[0]
        c = co2[(co2["padd"] == padd) & (co2["year"] <= o.year - 1)]
        co2_i = float(c.dropna(subset=["t_co2_per_kbbl"]).iloc[-1]["t_co2_per_kbbl"])
        rows.append({"district": d, "padd": padd, "cap_kbcd": cap_kbcd, "umax": umax,
                     "crude_usd_bbl": float(pr_prev[f"crude_{padd}"]),
                     "co2_t_per_kbbl": co2_i, "minrun": minrun,
                     "util_last12": float(util5.iloc[-12:].mean() / 100.0)})
    districts = pd.DataFrame(rows).set_index("district")

    # ---------------- demand ----------------
    dem = demand_panel.copy()
    meta = dem.drop_duplicates(["region", "product"]).set_index(["region", "product"])[["area", "hub", "kind"]]
    if forecast is None:
        dd = dem[dem["month"].isin(months)][["region", "product", "month", "kt"]]
    else:
        f = forecast[forecast["target_month"].isin(months)]
        dd = f.rename(columns={"product_name": "product", "target_month": "month", demand_col: "kt"})[
            ["region", "product", "month", "kt"]]
    dd = dd.join(meta, on=["region", "product"]).fillna({"kt": 0.0})
    dd = dd[dd["kt"] > 1e-6].reset_index(drop=True)

    # ---------------- hubs / inventory ----------------
    stocks = rc.hub_stock_panel(tables, dem)
    hub_rows = []
    for (h, k), g in stocks.groupby(["hub", "product"]):
        g = g.set_index("month")["kt"].sort_index()
        hist = g.loc[prev - pd.DateOffset(months=59):prev]
        if hist.empty:
            continue
        inv0 = float(hist.iloc[-1])
        end_m = months[-1].month
        same = hist[hist.index.month == end_m]
        target = float(same.mean() if len(same) else hist.mean()) * 0.95
        floor = float(hist.min()) * 0.90
        cap = float(hist.max()) * 1.10
        floor = min(floor, inv0 * 0.98)
        target = min(max(target, floor), cap)
        hub_rows.append({"hub": h, "product": k, "inv0": inv0, "floor": floor, "cap": max(cap, inv0),
                         "target": target})
    have = {(r["hub"], r["product"]) for r in hub_rows}
    for h in net.HUBS:  # pure transshipment where no stock series exists
        for k in PRODUCTS:
            if (h, k) not in have:
                hub_rows.append({"hub": h, "product": k, "inv0": 0.0, "floor": 0.0, "cap": 0.0, "target": 0.0})
    hubs = pd.DataFrame(hub_rows).set_index(["hub", "product"]).sort_index()

    # ---------------- lanes from observed inter-PADD movements ----------------
    mv = rc.movements_panel(tables)
    mv = mv[(mv["month"] <= prev) & (mv["month"] > prev - pd.DateOffset(months=lane_lookback))]
    lane_tot = mv.groupby(["from_area", "to_area", "mode", "month"])["kbbl"].sum().reset_index()
    lane_stats = lane_tot.groupby(["from_area", "to_area", "mode"])["kbbl"].agg(["mean", "max"]).reset_index()
    lane_rows, arc_rows = [], []
    for r in lane_stats.itertuples(index=False):
        if r.mean < 30 or (r.to_area, r.mode) not in net.MOVEMENT_DEST_HUBS:
            continue
        if r.from_area not in ("P1", "P2", "P3", "P4", "P5"):
            continue
        lane = f"{r.from_area}>{r.to_area}:{r.mode}"
        marine = r.mode == "marine"
        river = marine and (r.to_area == "P2" or r.from_area == "P2")
        lane_rows.append({"lane": lane, "from_padd": r.from_area, "to_area": r.to_area, "mode": r.mode,
                          "cap_kbbl": float(r.max) * 1.15, "mean_kbbl": float(r.mean), "marine": marine,
                          "river": bool(river)})
    lanes = pd.DataFrame(lane_rows).set_index("lane")

    def _dh_cost(mode, km, river=False):
        if mode == "pipeline" or mode == "intra":
            return C["pipeline_usd_per_tkm"] * km, net.EMISSION_G_PER_TKM["pipeline"] * km / 1e6
        if river:
            return 0.0, net.EMISSION_G_PER_TKM["barge"] * km / 1e6
        return 0.0, net.EMISSION_G_PER_TKM["short_sea"] * km / 1e6  # voyage cost via integer cargoes

    aid = 0
    # intra-PADD arcs
    for d, info in net.DISTRICT_INFO.items():
        padd = info[0]
        for h, hv in net.HUBS.items():
            area = hv[0]
            same = (padd == "P1" and area in ("1A", "1B", "1C")) or area == padd
            if not same:
                continue
            if padd == "P1" and area == "1C":
                continue  # no pipeline south out of the Delaware Valley / Appalachia
            if d == "AP" and area == "1A":
                continue
            km = net.road_km(net.district_coord(d), net.hub_coord(h))
            mode = "intra"
            if d == "EC" and h == "H1A":
                km, mode = net.SEA_KM[("P1", "H1A")], "barge_coastal"
            cost, em = _dh_cost("pipeline" if mode == "intra" else "barge", km)
            if mode == "barge_coastal":
                cost = 3.0 * 1.0  # ~$0.4/bbl short coastal barge, $/t
                em = net.EMISSION_G_PER_TKM["barge"] * km / 1e6
            arc_rows.append({"id": f"A{aid}", "d": d, "h": h, "mode": mode, "km": km, "lane": None,
                             "usd_per_t": cost, "co2_t_per_t": em})
            aid += 1
    # inter-PADD arcs
    for lane, L in lanes.iterrows():
        for d, info in net.DISTRICT_INFO.items():
            if info[0] != L["from_padd"]:
                continue
            if L["marine"] and not L["river"] and d not in net.MARINE_ORIGIN_DISTRICTS:
                continue  # inland refineries have no tanker berth
            if L["river"] and d not in net.RIVER_ORIGIN_DISTRICTS:
                continue
            for h in net.MOVEMENT_DEST_HUBS[(L["to_area"], L["mode"])]:
                if L["mode"] == "pipeline":
                    km = net.road_km(net.district_coord(d), net.hub_coord(h))
                elif L["river"]:
                    km = net.RIVER_KM.get((L["from_padd"], net.HUBS[h][0] if net.HUBS[h][0].startswith("P") else "P1"), 1700)
                else:
                    km = net.SEA_KM.get((L["from_padd"], h), 1.3 * net.road_km(net.district_coord(d), net.hub_coord(h)))
                cost, em = _dh_cost(L["mode"], km, L["river"])
                arc_rows.append({"id": f"A{aid}", "d": d, "h": h, "mode": L["mode"] + ("_river" if L["river"] else ""),
                                 "km": km, "lane": lane, "usd_per_t": cost, "co2_t_per_t": em})
                aid += 1
    arcs_dh = pd.DataFrame(arc_rows)
    lanes = lanes[lanes.index.isin(arcs_dh["lane"].dropna().unique())].copy()  # lanes nobody can load
    # lane cargo economics
    cargo, voyage = [], []
    for lane, L in lanes.iterrows():
        if not L["marine"]:
            cargo.append(np.nan); voyage.append(0.0); continue
        kms = arcs_dh[arcs_dh["lane"] == lane]["km"]
        km = float(kms.mean()) if len(kms) else 1500.0
        if L["river"]:
            cargo.append(C["barge_tow_kbbl"]); voyage.append(net.barge_tow_usd(km) / 1000.0)
        else:
            cargo.append(C["tanker_cargo_kbbl"]); voyage.append(net.tanker_voyage_usd(km) / 1000.0)
    lanes["cargo_kbbl"] = cargo
    lanes["voyage_kusd"] = voyage

    # ---------------- direct rack (district -> nearby states) ----------------
    regions = dd.drop_duplicates("region")[["region", "area", "hub", "kind"]]
    dr = []
    for d in net.DISTRICT_INFO:
        for r in regions.itertuples(index=False):
            if r.kind != "state":
                continue
            km = net.road_km(net.district_coord(d), net.state_coord(r.region))
            if km <= 450:
                dr.append({"id": f"R{len(dr)}", "d": d, "r": r.region, "km": km,
                           "usd_per_t": C["truck_usd_per_tkm"] * km + C["truck_fixed_usd_per_t"],
                           "co2_t_per_t": net.EMISSION_G_PER_TKM["road"] * km / 1e6})
    arcs_dr = pd.DataFrame(dr)

    # ---------------- hub -> region ----------------
    hr = []
    last_mile_cost = C["truck_usd_per_tkm"] * C["last_mile_km"] + C["truck_fixed_usd_per_t"]
    last_mile_co2 = net.EMISSION_G_PER_TKM["road"] * C["last_mile_km"] / 1e6
    for r in regions.itertuples(index=False):
        if r.kind == "pool":
            cands = net.hubs_of_area(r.area)
        else:
            cands = {r.hub}
            for h in net.HUBS:
                if net.road_km(net.state_coord(r.region), net.hub_coord(h)) <= 900:
                    cands.add(h)
        for h in sorted(cands):
            if r.kind == "pool":
                km = 300.0
            else:
                km = net.road_km(net.state_coord(r.region), net.hub_coord(h))
            if r.region in ("HI", "AK"):
                key = ("H5", "HI") if r.region == "HI" else ("H5N", "AK")
                if h != key[0]:
                    continue
                km = net.SEA_KM[key]
                cost = net.tanker_voyage_usd(km) / (C["tanker_cargo_kbbl"] * 1000 * 0.1185) + last_mile_cost
                em = net.EMISSION_G_PER_TKM["short_sea"] * km / 1e6 + last_mile_co2
            elif r.kind == "state" and r.region in net.MARINE_SUPPLIED_STATES and h != r.hub:
                # no pipeline into the state: a distant hub can only truck product in
                cost = C["truck_usd_per_tkm"] * km + C["truck_fixed_usd_per_t"]
                em = net.EMISSION_G_PER_TKM["road"] * km / 1e6
            else:
                cost = C["pipeline_usd_per_tkm"] * km + last_mile_cost
                em = net.EMISSION_G_PER_TKM["pipeline"] * km / 1e6 + last_mile_co2
            hr.append({"id": f"H{len(hr)}", "h": h, "r": r.region, "km": km, "usd_per_t": cost, "co2_t_per_t": em})
    arcs_hr = pd.DataFrame(hr)

    # ---------------- trade ----------------
    tr = rc.trade_panel(tables)
    tr = tr[(tr["month"] <= prev) & (tr["month"] > prev - pd.DateOffset(months=24))]
    us_int = float(rc.refinery_co2_intensity(tables).query("padd=='US' and year <= @o.year-1")
                   .dropna().iloc[-1]["t_co2_per_kbbl"])
    mass_yield = float(yl.mean().sum())  # kt product per kbbl crude (U.S. average of modelled products)
    foreign_ref_co2 = us_int / 1000.0 / mass_yield  # t CO2 per t product
    hub_dem = dd.groupby(["hub", "product"])["kt"].sum()
    imp_rows = []
    for padd in ["P1", "P2", "P3", "P4", "P5"]:
        hubs_p = {"P1": ["H1A", "H1B", "H1F"], "P2": ["H2N"], "P3": ["H3"], "P4": ["H4"],
                  "P5": ["H5", "H5N"]}[padd]
        for k in PRODUCTS:
            s = tr[(tr["padd"] == padd) & (tr["flow"] == "imports") & (tr["product"] == k)]
            if s.empty or s["kt"].max() < 1:
                continue
            capk = float(s.groupby("month")["kt"].sum().max()) * 1.2
            if k == "GAS":  # imported blendstock + terminal ethanol, finished-gasoline equivalent
                capk *= float(yl.attrs.get("gas_uplift", 1.0))
            w = np.array([hub_dem.get((h, k), 0.0) for h in hubs_p]) + 1e-6
            w = w / w.sum()
            for h, share in zip(hubs_p, w):
                marine = h in net.IMPORT_HUBS
                km = net.SEA_KM.get(("IMPORT", h), 1000.0)
                reg = rc.hub_price_region(h)
                em = (net.EMISSION_G_PER_TKM["deep_sea_tanker"] if marine else net.EMISSION_G_PER_TKM["pipeline"]) * km / 1e6
                imp_rows.append({"hub": h, "product": k, "cap_kt": capk * share,
                                 "usd_per_t": spot[(reg, k)], "co2_t_per_t": em + foreign_ref_co2,
                                 "marine": marine})
    imports = pd.DataFrame(imp_rows)
    exp_rows = []
    tr12 = tr[tr["month"] > prev - pd.DateOffset(months=12)]
    for padd, h in net.EXPORT_HUBS.items():
        for k in PRODUCTS:
            s = tr12[(tr12["padd"] == padd) & (tr12["flow"] == "exports") & (tr12["product"] == k)]
            commit = float(s.groupby("month")["kt"].sum().mean()) if len(s) else 0.0
            if commit < 1:
                continue
            for mth in months:
                exp_rows.append({"hub": h, "product": k, "month": mth, "commit_kt": commit,
                                 "spot_usd_per_t": spot[(rc.hub_price_region(h), k)]})
    exports = pd.DataFrame(exp_rows)
    # spare export-terminal capacity for surplus co-products, sold at a netback
    xcap_rows = []
    tr24 = tr
    for padd, h in net.EXPORT_HUBS.items():
        if h not in net.IMPORT_HUBS:  # inland hubs: no marine export terminal
            continue
        for k in PRODUCTS:
            s = tr24[(tr24["padd"] == padd) & (tr24["flow"] == "exports") & (tr24["product"] == k)]
            if s.empty:
                continue
            mx = float(s.groupby("month")["kt"].max().max()) * 1.2
            commit = float(exports[(exports["hub"] == h) & (exports["product"] == k)]["commit_kt"].max()) \
                if len(exports) and ((exports["hub"] == h) & (exports["product"] == k)).any() else 0.0
            spare = max(0.0, mx - commit)
            if spare > 1:
                xcap_rows.append({"hub": h, "product": k, "cap_kt": spare,
                                  "netback_usd_per_t": 0.95 * spot[(rc.hub_price_region(h), k)]})
    export_spare = pd.DataFrame(xcap_rows)

    # ---------------- gas-plant propane ----------------
    ngl = rc.ngl_propane_supply(tables)
    ngl = ngl[(ngl["month"] <= prev) & (ngl["month"] > prev - pd.DateOffset(months=12))]
    ngl_hub = {"P1": "H1B", "P2": "H2S", "P3": "H3", "P4": "H4", "P5": "H5"}
    ngl_rows = []
    for padd, g in ngl.groupby("padd"):
        capk = float(g["kt"].mean()) * 1.10
        for mth in months:
            ngl_rows.append({"hub": ngl_hub[padd], "month": mth, "cap_kt": capk,
                             "usd_per_t": spot[(rc.hub_price_region(ngl_hub[padd]), "LPG")],
                             "co2_t_per_t": 0.05})
    ngl_df = pd.DataFrame(ngl_rows)

    # ---------------- risk ----------------
    risk_df = None
    if risk is not None:
        sev = {}
        if labels is not None:
            past = labels[(labels["month"] <= prev) & (labels["outage"] == 1)]
            for d in net.DISTRICT_INFO:
                s = past[past["district"] == d]["severity"]
                sev[d] = float(s.mean()) if len(s) else 0.15
        r = risk[risk["target_month"].isin(months)][["district", "target_month", "p"]].rename(
            columns={"target_month": "month"})
        r["sev"] = r["district"].map(sev).fillna(0.15)
        risk_df = r

    params = {"holding_kusd_per_kt": {k: C["holding_usd_per_bbl_month"] / KT_PER_KBBL[k] for k in PRODUCTS},
              "opex_usd_bbl": C["refining_opex_usd_per_bbl"],
              "shortage_mult": C["shortage_multiplier"], "export_mult": C["export_shortfall_multiplier"],
              "stock_mult": C["stock_target_multiplier"],
              "cargo_kbbl": C["tanker_cargo_kbbl"], "gas_uplift": yl.attrs.get("gas_uplift")}
    return PlanningInputs(origin=o, months=months, districts=districts, yields=yl, hubs=hubs, demand=dd,
                          arcs_dh=arcs_dh, arcs_dr=arcs_dr, arcs_hr=arcs_hr, lanes=lanes, imports=imports,
                          exports=exports, ngl=ngl_df, spot=spot, risk=risk_df, params=params,
                          export_spare=export_spare)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class PlanResult:
    status: str
    objective_kusd: float
    kpis: Dict[str, float]
    crude: pd.DataFrame
    flows_dh: pd.DataFrame
    flows_dr: pd.DataFrame
    flows_hr: pd.DataFrame
    inventory: pd.DataFrame
    imports: pd.DataFrame
    shortages: pd.DataFrame
    cargoes: pd.DataFrame
    solve_s: float
    mip_gap: Optional[float] = None
    n_int: int = 0
    n_vars: int = 0
    n_cons: int = 0


def _cap_eff(inp: PlanningInputs, d: str, t: int, risk_aware: bool) -> float:
    row = inp.districts.loc[d]
    days = inp.months[t].days_in_month
    cap = row["cap_kbcd"] * days * row["umax"]
    if risk_aware and inp.risk is not None:
        r = inp.risk[(inp.risk["district"] == d) & (inp.risk["month"] == inp.months[t])]
        if len(r):
            cap *= (1.0 - float(r["p"].iloc[0]) * float(r["sev"].iloc[0]))
    return cap


def build_model(inp: PlanningInputs, carbon_kusd_per_kt: float = 0.0, risk_aware: bool = False,
                fixed: Optional[Dict] = None, integer: bool = True,
                availability: Optional[Dict[Tuple[str, int], float]] = None,
                emergency_imports: bool = False) -> pyo.ConcreteModel:
    """Build the Pyomo model. `fixed` (second-stage evaluation) fixes crude runs
    within a recourse band, marine cargoes and import cargoes from a plan."""
    m = pyo.ConcreteModel("OptiGreenChem")
    T = inp.T
    D = list(inp.districts.index)
    K = PRODUCTS
    H = sorted(set(h for h, _ in inp.hubs.index))
    dem = inp.demand
    R = sorted(dem["region"].unique())
    A = inp.arcs_dh.set_index("id")
    G = inp.arcs_dr.set_index("id") if len(inp.arcs_dr) else pd.DataFrame(columns=["d", "r", "usd_per_t", "co2_t_per_t"])
    Q = inp.arcs_hr.set_index("id")
    L = inp.lanes
    Y = inp.yields
    P = inp.params
    dem_idx = {(r.region, r.product, inp.months.index(r.month)): r.kt for r in dem.itertuples(index=False)}
    area_of = dem.drop_duplicates("region").set_index("region")["area"].to_dict()

    m.x = pyo.Var(D, T, domain=pyo.NonNegativeReals)
    m.f = pyo.Var(list(A.index), K, T, domain=pyo.NonNegativeReals)
    m.g = pyo.Var(list(G.index), K, T, domain=pyo.NonNegativeReals)
    m.q = pyo.Var(list(Q.index), K, T, domain=pyo.NonNegativeReals)
    hk = list(inp.hubs.index)
    m.inv = pyo.Var(hk, T, domain=pyo.NonNegativeReals)
    imp_idx = [(r.hub, r.product) for r in inp.imports.itertuples(index=False)]
    m.imp = pyo.Var(imp_idx, T, domain=pyo.NonNegativeReals)
    ngl_h = sorted(inp.ngl["hub"].unique()) if len(inp.ngl) else []
    m.ngl = pyo.Var(ngl_h, T, domain=pyo.NonNegativeReals)
    m.xs = pyo.Var(hk, T, domain=pyo.NonNegativeReals)
    # distressed disposal (sale into other markets at a loss) keeps the recourse model
    # feasible when fixed crude runs / contracted imports exceed what can be stored or sold
    m.dump = pyo.Var(hk, T, domain=pyo.NonNegativeReals)
    m.ts = pyo.Var(hk, domain=pyo.NonNegativeReals)  # shortfall vs. terminal stock target
    exp_idx = sorted(set((r.hub, r.product) for r in inp.exports.itertuples(index=False)))
    m.ex = pyo.Var(exp_idx, T, domain=pyo.NonNegativeReals)
    m.es = pyo.Var(exp_idx, T, domain=pyo.NonNegativeReals)
    dem_keys = list(dem_idx.keys())
    m.u = pyo.Var(dem_keys, domain=pyo.NonNegativeReals)
    marine_lanes = [l for l in L.index if L.loc[l, "marine"]]
    dom_int = pyo.NonNegativeIntegers if integer else pyo.NonNegativeReals
    m.n = pyo.Var(marine_lanes, T, domain=dom_int)
    port_imp_hubs = sorted(set(h for h, k in imp_idx if h in net.IMPORT_HUBS))
    m.mc = pyo.Var(port_imp_hubs, T, domain=dom_int)
    if emergency_imports:
        m.eimp = pyo.Var(imp_idx, T, domain=pyo.NonNegativeReals)

    # ---- crude run bounds ----
    m.cap = pyo.ConstraintList()
    for d in D:
        for t in T:
            hi = _cap_eff(inp, d, t, risk_aware)
            lo = inp.districts.loc[d, "minrun"] * hi
            if availability is not None:
                hi = min(hi, availability[(d, t)])
                lo = min(lo, hi)
            if fixed is not None:
                xp = fixed["x"][(d, t)]
                hi_f = min(1.05 * xp, availability[(d, t)] if availability else 1.05 * xp)
                lo = min(0.90 * xp, hi_f)
                hi = hi_f
            m.cap.add(m.x[d, t] <= hi)
            m.cap.add(m.x[d, t] >= lo)

    # ---- district product balance ----
    out_f = {d: list(A.index[A["d"] == d]) for d in D}
    out_g = {d: list(G.index[G["d"] == d]) for d in D} if len(G) else {d: [] for d in D}
    m.dbal = pyo.ConstraintList()
    for d in D:
        for k in K:
            yk = float(Y.loc[d, k]) if d in Y.index else 0.0
            for t in T:
                m.dbal.add(yk * m.x[d, t] == sum(m.f[a, k, t] for a in out_f[d]) + sum(m.g[a, k, t] for a in out_g[d]))

    # ---- hub balance and inventory ----
    in_f = {h: list(A.index[A["h"] == h]) for h in H}
    out_q = {h: list(Q.index[Q["h"] == h]) for h in H}
    exp_lookup = {(r.hub, r.product, inp.months.index(r.month)): r.commit_kt for r in inp.exports.itertuples(index=False)}
    ngl_cap = {(r.hub, inp.months.index(r.month)): r.cap_kt for r in inp.ngl.itertuples(index=False)}
    m.hbal = pyo.ConstraintList()
    m.invb = pyo.ConstraintList()
    for (h, k) in hk:
        hv = inp.hubs.loc[(h, k)]
        for t in T:
            prev = hv["inv0"] if t == 0 else m.inv[(h, k), t - 1]
            inflow = sum(m.f[a, k, t] for a in in_f.get(h, []))
            if (h, k) in imp_idx:
                inflow = inflow + m.imp[(h, k), t]
                if emergency_imports:
                    inflow = inflow + m.eimp[(h, k), t]
            if k == "LPG" and h in ngl_h:
                inflow = inflow + m.ngl[h, t]
            outflow = sum(m.q[a, k, t] for a in out_q.get(h, [])) + m.xs[(h, k), t] + m.dump[(h, k), t]
            if (h, k) in exp_idx:
                outflow = outflow + m.ex[(h, k), t]
            m.hbal.add(m.inv[(h, k), t] == prev + inflow - outflow)
            m.invb.add(m.inv[(h, k), t] >= hv["floor"])
            m.invb.add(m.inv[(h, k), t] <= hv["cap"])
        m.invb.add(m.inv[(h, k), T[-1]] + m.ts[(h, k)] >= hv["target"])

    # ---- surplus exports: only through marine export terminals, within spare capacity ----
    xsp = {}
    if inp.export_spare is not None and len(inp.export_spare):
        xsp = {(r.hub, r.product): (r.cap_kt, r.netback_usd_per_t) for r in inp.export_spare.itertuples(index=False)}
    m.xsc = pyo.ConstraintList()
    for (h, k) in hk:
        for t in T:
            if (h, k) in xsp:
                m.xsc.add(m.xs[(h, k), t] <= xsp[(h, k)][0])
            elif h in net.IMPORT_HUBS:
                m.xs[(h, k), t].fix(0.0)
            # inland hubs keep a zero-value outlet (sales to other markets) so joint
            # co-products never make the model infeasible

    # ---- exports ----
    m.expc = pyo.ConstraintList()
    for (h, k) in exp_idx:
        for t in T:
            m.expc.add(m.ex[(h, k), t] + m.es[(h, k), t] == exp_lookup.get((h, k, t), 0.0))

    # ---- NGL ----
    m.nglc = pyo.ConstraintList()
    for h in ngl_h:
        for t in T:
            m.nglc.add(m.ngl[h, t] <= ngl_cap.get((h, t), 0.0))

    # ---- demand ----
    in_q = {}
    for a, r in Q.iterrows():
        in_q.setdefault(r["r"], []).append(a)
    in_g = {}
    for a, r in G.iterrows():
        in_g.setdefault(r["r"], []).append(a)
    m.dem = pyo.ConstraintList()
    for (r, k, t), v in dem_idx.items():
        m.dem.add(sum(m.q[a, k, t] for a in in_q.get(r, [])) + sum(m.g[a, k, t] for a in in_g.get(r, []))
                  + m.u[(r, k, t)] == v)
    # no deliveries of a product to a region without demand for it
    m.nodem = pyo.ConstraintList()
    dem_rk = set((r, k) for (r, k, _t) in dem_idx)
    for a, rr in Q.iterrows():
        for k in K:
            if (rr["r"], k) not in dem_rk:
                for t in T:
                    m.q[a, k, t].fix(0.0)
    for a, rr in G.iterrows():
        for k in K:
            if (rr["r"], k) not in dem_rk:
                for t in T:
                    m.g[a, k, t].fix(0.0)

    # ---- lanes and marine cargoes ----
    lane_arcs = {l: list(A.index[A["lane"] == l]) for l in L.index}
    m.lane = pyo.ConstraintList()
    for l in L.index:
        for t in T:
            # gasoline moves as blendstock (BOB); ethanol/NGL are added at the terminal
            vol = sum(m.f[a, k, t] / KT_PER_KBBL[k] / (P["gas_uplift"] if k == "GAS" else 1.0)
                      for a in lane_arcs[l] for k in K)
            m.lane.add(vol <= float(L.loc[l, "cap_kbbl"]))
            if L.loc[l, "marine"]:
                m.lane.add(vol <= float(L.loc[l, "cargo_kbbl"]) * m.n[l, t])
                if fixed is not None:  # chartered cargoes are a sunk commitment
                    m.lane.add(m.n[l, t] == fixed["n"][(l, t)])

    # ---- imports ----
    imp_cap = {(r.hub, r.product): r.cap_kt for r in inp.imports.itertuples(index=False)}
    m.impc = pyo.ConstraintList()
    for (h, k) in imp_idx:
        for t in T:
            if fixed is not None:  # contracted import volumes are taken
                m.impc.add(m.imp[(h, k), t] == fixed["imp"][((h, k), t)])
            else:
                m.impc.add(m.imp[(h, k), t] <= imp_cap[(h, k)])
            if emergency_imports:
                m.impc.add(m.eimp[(h, k), t] <= 0.5 * imp_cap[(h, k)])
    for h in port_imp_hubs:
        for t in T:
            vol = sum(m.imp[(hh, k), t] / KT_PER_KBBL[k] / (P["gas_uplift"] if k == "GAS" else 1.0)
                      for (hh, k) in imp_idx if hh == h)
            m.impc.add(vol <= P["cargo_kbbl"] * m.mc[h, t])
            if fixed is not None:
                m.impc.add(m.mc[h, t] == fixed["mc"][(h, t)])

    # ---- objective ----
    imp_price = {(r.hub, r.product): r.usd_per_t for r in inp.imports.itertuples(index=False)}
    imp_co2 = {(r.hub, r.product): r.co2_t_per_t for r in inp.imports.itertuples(index=False)}
    ngl_price = {(r.hub, inp.months.index(r.month)): r.usd_per_t for r in inp.ngl.itertuples(index=False)}
    ngl_co2 = {(r.hub, inp.months.index(r.month)): r.co2_t_per_t for r in inp.ngl.itertuples(index=False)}
    exp_spot = {(r.hub, r.product): r.spot_usd_per_t for r in inp.exports.itertuples(index=False)}

    def short_pen(r, k):
        return P["shortage_mult"] * inp.spot[(_area_price_region(area_of[r]), k)]

    e = {}
    e["crude"] = sum((inp.districts.loc[d, "crude_usd_bbl"] + P["opex_usd_bbl"]) * m.x[d, t] for d in D for t in T)
    e["transport_pipe"] = sum(A.loc[a, "usd_per_t"] * m.f[a, k, t] for a in A.index for k in K for t in T)
    e["transport_rack"] = (sum(G.loc[a, "usd_per_t"] * m.g[a, k, t] for a in G.index for k in K for t in T)
                           + sum(Q.loc[a, "usd_per_t"] * m.q[a, k, t] for a in Q.index for k in K for t in T))
    e["voyages"] = sum(float(L.loc[l, "voyage_kusd"]) * m.n[l, t] for l in marine_lanes for t in T)
    e["imports"] = sum(imp_price[i] * m.imp[i, t] for i in imp_idx for t in T)
    if emergency_imports:
        e["imports"] = e["imports"] + sum((1 + net.COST_ASSUMPTIONS["emergency_import_premium"]) * imp_price[i]
                                          * m.eimp[i, t] for i in imp_idx for t in T)
    e["ngl"] = sum(ngl_price[(h, t)] * m.ngl[h, t] for h in ngl_h for t in T)
    e["holding"] = sum(P["holding_kusd_per_kt"][k] * 1000 / 1000 * m.inv[(h, k), t] for (h, k) in hk for t in T)
    e["shortage_penalty"] = sum(short_pen(r, k) * m.u[(r, k, t)] for (r, k, t) in dem_keys)
    e["export_penalty"] = sum(P["export_mult"] * exp_spot[i] * m.es[i, t] for i in exp_idx for t in T)
    e["stock_target_penalty"] = sum(P["stock_mult"] * inp.spot[("gulf", k)] * m.ts[(h, k)] for (h, k) in hk)
    e["export_revenue"] = -sum(xsp[(h, k)][1] * m.xs[(h, k), t] for (h, k) in xsp for t in T)
    e["distressed_disposal"] = sum(P.get("dump_usd_per_t", 150.0) * m.dump[(h, k), t] for (h, k) in hk for t in T)
    co2 = {}
    co2["refining"] = sum(inp.districts.loc[d, "co2_t_per_kbbl"] / 1000.0 * m.x[d, t] for d in D for t in T)
    co2["transport"] = (sum(A.loc[a, "co2_t_per_t"] * m.f[a, k, t] for a in A.index for k in K for t in T)
                        + sum(G.loc[a, "co2_t_per_t"] * m.g[a, k, t] for a in G.index for k in K for t in T)
                        + sum(Q.loc[a, "co2_t_per_t"] * m.q[a, k, t] for a in Q.index for k in K for t in T))
    co2["imports"] = sum(imp_co2[i] * m.imp[i, t] for i in imp_idx for t in T)
    if emergency_imports:
        co2["imports"] = co2["imports"] + sum(imp_co2[i] * m.eimp[i, t] for i in imp_idx for t in T)
    co2["ngl"] = sum(ngl_co2[(h, t)] * m.ngl[h, t] for h in ngl_h for t in T)
    m.cost_expr = e
    m.co2_expr = co2
    total_cost = sum(e.values())
    total_co2 = sum(co2.values())
    m.obj = pyo.Objective(expr=total_cost + carbon_kusd_per_kt * total_co2, sense=pyo.minimize)
    m._meta = dict(D=D, T=T, K=K, H=H, hk=hk, imp_idx=imp_idx, ngl_h=ngl_h, exp_idx=exp_idx,
                   dem_keys=dem_keys, dem_idx=dem_idx, marine_lanes=marine_lanes, port_imp_hubs=port_imp_hubs,
                   A=A, G=G, Q=Q, emergency=emergency_imports)
    return m


def solve(m: pyo.ConcreteModel, time_limit: float = 60.0, mip_gap: float = 0.002) -> Tuple[str, float, Optional[float]]:
    opt = pyo.SolverFactory("highs")
    try:
        if not opt.available(exception_flag=False):
            raise RuntimeError
    except Exception:  # older Pyomo releases expose HiGHS as "appsi_highs"
        opt = pyo.SolverFactory("appsi_highs")
    opt.options["time_limit"] = time_limit
    opt.options["mip_rel_gap"] = mip_gap
    opt.options["log_to_console"] = False
    t0 = time.time()
    try:
        res = opt.solve(m, tee=False, load_solutions=True)
    except Exception as exc:  # no feasible solution found
        return f"failed: {type(exc).__name__}", time.time() - t0, None
    el = time.time() - t0
    tc = str(res.solver.termination_condition)
    gap = None
    try:
        lb = res.problem.lower_bound
        ub = res.problem.upper_bound
        if lb is not None and ub is not None and abs(ub) > 1e-9:
            gap = abs(ub - lb) / abs(ub)
    except Exception:
        pass
    return tc, el, gap


def extract(m: pyo.ConcreteModel, inp: PlanningInputs, status: str, el: float, gap) -> PlanResult:
    M = m._meta
    v = pyo.value
    if status.startswith("failed"):
        e = pd.DataFrame()
        return PlanResult(status=status, objective_kusd=float("nan"), kpis={"fill_rate": float("nan")},
                          crude=e, flows_dh=e, flows_dr=e, flows_hr=e, inventory=e, imports=e, shortages=e,
                          cargoes=e, solve_s=el)
    T, K = M["T"], M["K"]
    mon = inp.months
    crude = pd.DataFrame([{"district": d, "month": mon[t], "crude_kbbl": v(m.x[d, t]),
                           "cap_kbbl": inp.districts.loc[d, "cap_kbcd"] * mon[t].days_in_month}
                          for d in M["D"] for t in T])
    crude["utilization"] = crude["crude_kbbl"] / crude["cap_kbbl"]
    A, G, Q = M["A"], M["G"], M["Q"]
    fdh = pd.DataFrame([{"arc": a, "district": A.loc[a, "d"], "hub": A.loc[a, "h"], "mode": A.loc[a, "mode"],
                         "lane": A.loc[a, "lane"], "product": k, "month": mon[t], "kt": v(m.f[a, k, t])}
                        for a in A.index for k in K for t in T])
    fdh = fdh[fdh["kt"] > 1e-6]
    fdr = pd.DataFrame([{"district": G.loc[a, "d"], "region": G.loc[a, "r"], "product": k, "month": mon[t],
                         "kt": v(m.g[a, k, t])} for a in G.index for k in K for t in T]) if len(G) else pd.DataFrame()
    if len(fdr):
        fdr = fdr[fdr["kt"] > 1e-6]
    fhr = pd.DataFrame([{"hub": Q.loc[a, "h"], "region": Q.loc[a, "r"], "product": k, "month": mon[t],
                         "kt": v(m.q[a, k, t])} for a in Q.index for k in K for t in T])
    fhr = fhr[fhr["kt"] > 1e-6]
    inv = pd.DataFrame([{"hub": h, "product": k, "month": mon[t], "kt": v(m.inv[(h, k), t]),
                         "floor": inp.hubs.loc[(h, k), "floor"], "cap": inp.hubs.loc[(h, k), "cap"]}
                        for (h, k) in M["hk"] for t in T])
    imp = pd.DataFrame([{"hub": h, "product": k, "month": mon[t], "kt": v(m.imp[(h, k), t])
                         + (v(m.eimp[(h, k), t]) if M["emergency"] else 0.0),
                         "emergency_kt": v(m.eimp[(h, k), t]) if M["emergency"] else 0.0}
                        for (h, k) in M["imp_idx"] for t in T])
    sh = pd.DataFrame([{"region": r, "product": k, "month": mon[t], "demand_kt": M["dem_idx"][(r, k, t)],
                        "short_kt": v(m.u[(r, k, t)])} for (r, k, t) in M["dem_keys"]])
    cg = pd.DataFrame([{"lane": l, "month": mon[t], "cargoes": round(v(m.n[l, t]))} for l in M["marine_lanes"] for t in T]
                      + [{"lane": f"IMPORT@{h}", "month": mon[t], "cargoes": round(v(m.mc[h, t]))}
                         for h in M["port_imp_hubs"] for t in T])
    cost = {k: float(v(e)) for k, e in m.cost_expr.items()}
    co2 = {k: float(v(e)) for k, e in m.co2_expr.items()}
    tot_dem = sh["demand_kt"].sum()
    kpis = {f"cost_{k}_musd": c / 1000.0 for k, c in cost.items()}
    kpis.update({f"co2_{k}_kt": c for k, c in co2.items()})
    kpis["cost_total_musd"] = sum(cost.values()) / 1000.0
    kpis["cost_supply_chain_musd"] = (sum(cost.values()) - cost["shortage_penalty"] - cost["export_penalty"]
                                      - cost["stock_target_penalty"]) / 1000.0
    kpis["surplus_exports_kt"] = float(sum(v(m.xs[i, t]) for i in M["hk"] for t in T
                                           if i[0] in net.IMPORT_HUBS))
    kpis["cost_logistics_musd"] = (cost["transport_pipe"] + cost["transport_rack"] + cost["voyages"]
                                   + cost["holding"]) / 1000.0
    kpis["co2_total_kt"] = sum(co2.values())
    kpis["demand_kt"] = tot_dem
    kpis["unmet_kt"] = sh["short_kt"].sum()
    kpis["fill_rate"] = 1.0 - kpis["unmet_kt"] / tot_dem if tot_dem > 0 else 1.0
    for k in K:
        s = sh[sh["product"] == k]
        kpis[f"fill_{k}"] = 1.0 - s["short_kt"].sum() / s["demand_kt"].sum() if s["demand_kt"].sum() > 0 else 1.0
    kpis["crude_kbbl"] = crude["crude_kbbl"].sum()
    kpis["imports_kt"] = imp["kt"].sum()
    kpis["emergency_imports_kt"] = imp["emergency_kt"].sum()
    kpis["export_short_kt"] = float(sum(v(m.es[i, t]) for i in M["exp_idx"] for t in T))
    kpis["distressed_disposal_kt"] = float(sum(v(m.dump[i, t]) for i in M["hk"] for t in T))
    kpis["marine_cargoes"] = int(cg["cargoes"].sum()) if len(cg) else 0
    kpis["co2_intensity_kg_per_t"] = kpis["co2_total_kt"] / max(tot_dem - kpis["unmet_kt"], 1e-9) * 1000
    nvars = sum(1 for _ in m.component_data_objects(pyo.Var, active=True))
    nint = sum(1 for vv in m.component_data_objects(pyo.Var, active=True) if vv.is_integer())
    ncons = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
    return PlanResult(status=status, objective_kusd=float(v(m.obj)), kpis=kpis, crude=crude, flows_dh=fdh,
                      flows_dr=fdr, flows_hr=fhr, inventory=inv, imports=imp, shortages=sh, cargoes=cg,
                      solve_s=el, mip_gap=gap, n_int=nint, n_vars=nvars, n_cons=ncons)


def plan(inp: PlanningInputs, carbon_kusd_per_kt: float = 0.0, risk_aware: bool = False,
         time_limit: float = 60.0, mip_gap: float = 0.002) -> PlanResult:
    m = build_model(inp, carbon_kusd_per_kt=carbon_kusd_per_kt, risk_aware=risk_aware)
    status, el, gap = solve(m, time_limit, mip_gap)
    res = extract(m, inp, status, el, gap)
    res.model = m
    return res


def first_stage(m: pyo.ConcreteModel) -> Dict:
    M = m._meta
    v = pyo.value
    return {"x": {(d, t): v(m.x[d, t]) for d in M["D"] for t in M["T"]},
            "n": {(l, t): round(v(m.n[l, t])) for l in M["marine_lanes"] for t in M["T"]},
            "imp": {(i, t): v(m.imp[i, t]) for i in M["imp_idx"] for t in M["T"]},
            "mc": {(h, t): round(v(m.mc[h, t])) for h in M["port_imp_hubs"] for t in M["T"]}}


def evaluate(inp_actual: PlanningInputs, fs: Dict, availability: Dict[Tuple[str, int], float],
             carbon_kusd_per_kt: float = 0.0, time_limit: float = 60.0) -> PlanResult:
    """Second stage: actual demand and actual refinery availability, first-stage
    decisions (crude runs within -10 %/+5 %, marine cargoes, import cargoes)
    fixed from the plan; pipelines, deliveries and inventories re-optimised;
    emergency imports allowed at a 25 % premium."""
    m = build_model(inp_actual, carbon_kusd_per_kt=carbon_kusd_per_kt, risk_aware=False, fixed=fs,
                    integer=False, availability=availability, emergency_imports=True)
    status, el, gap = solve(m, time_limit, 1e-6)
    return extract(m, inp_actual, status, el, gap)


def actual_availability(inp: PlanningInputs, labels: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """Realised refinery availability (kbbl): normal months = capacity x u_max;
    outage months = capacity x actual utilisation."""
    out = {}
    for d in inp.districts.index:
        for t, mth in enumerate(inp.months):
            row = inp.districts.loc[d]
            cap = row["cap_kbcd"] * mth.days_in_month
            lab = labels[(labels["district"] == d) & (labels["month"] == mth)]
            if len(lab) and lab["outage"].iloc[0] == 1:
                out[(d, t)] = cap * float(lab["utilization_pct"].iloc[0]) / 100.0
            else:
                out[(d, t)] = cap * row["umax"]
    return out


def bau_first_stage(inp: PlanningInputs, tables: Dict[str, pd.DataFrame]) -> Dict:
    """Business-as-usual rule: each district runs at its trailing-12-month mean
    utilisation; marine cargoes and imports repeat the trailing-12-month average
    pattern. Same recourse evaluation as the optimised plans."""
    prev = inp.origin - pd.DateOffset(months=1)
    x = {(d, t): inp.districts.loc[d, "cap_kbcd"] * inp.months[t].days_in_month * inp.districts.loc[d, "util_last12"]
         for d in inp.districts.index for t in inp.T}
    mv = rc.movements_panel(tables)
    mv = mv[(mv["month"] <= prev) & (mv["month"] > prev - pd.DateOffset(months=12))]
    n = {}
    for l, L in inp.lanes.iterrows():
        if not L["marine"]:
            continue
        s = mv[(mv["from_area"] == L["from_padd"]) & (mv["to_area"] == L["to_area"]) & (mv["mode"] == L["mode"])]
        avg = s.groupby("month")["kbbl"].sum().mean() if len(s) else 0.0
        for t in inp.T:
            n[(l, t)] = int(np.ceil(avg / L["cargo_kbbl"])) if avg > 0 else 0
    tr = rc.trade_panel(tables)
    tr = tr[(tr["month"] <= prev) & (tr["month"] > prev - pd.DateOffset(months=12)) & (tr["flow"] == "imports")]
    imp, mc = {}, {}
    caps = {(r.hub, r.product): r.cap_kt for r in inp.imports.itertuples(index=False)}
    for (h, k), capk in caps.items():
        padd = net.HUBS[h][0] if net.HUBS[h][0].startswith("P") else "P1"
        s = tr[(tr["padd"] == padd) & (tr["product"] == k)].groupby("month")["kt"].sum()
        share = capk / max(sum(c for (hh, kk), c in caps.items() if kk == k and
                               (net.HUBS[hh][0] if net.HUBS[hh][0].startswith("P") else "P1") == padd), 1e-9)
        for t in inp.T:
            imp[((h, k), t)] = float(s.mean()) * share if len(s) else 0.0
    for h in sorted(set(h for (h, k) in caps if h in net.IMPORT_HUBS)):
        for t in inp.T:
            vol = sum(imp[((hh, k), t)] / KT_PER_KBBL[k] for (hh, k) in caps if hh == h)
            mc[(h, t)] = int(np.ceil(vol / net.COST_ASSUMPTIONS["tanker_cargo_kbbl"]))
    return {"x": x, "n": n, "imp": imp, "mc": mc}
