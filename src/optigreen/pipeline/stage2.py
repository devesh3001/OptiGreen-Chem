"""
OptiGreen-Chem Stage-2 pipeline on real EIA data.

    python scripts/run_stage2.py --bulk PET.zip            # full run (~35-45 min on 2 CPUs)
    python scripts/run_stage2.py --quick                   # 6 origins, smoke test

Steps
-----
1. data      : PET.zip -> tidy tables -> demand / district / stock panels
2. forecast  : quantile XGBoost, refit every January on data before that year,
               evaluated on 2018-2021 rolling origins (h = 1..3) vs baselines
3. risk      : outage models (climatology, logistic, XGBoost, GAT, GAT self-loop)
               on a fixed chronological split + annual refits for planning
4. backtest  : every month 2018-01 .. 2021-12, plan 3 months ahead with each
               strategy, then evaluate the plan against ACTUAL demand and ACTUAL
               refinery availability (two-stage recourse model)
5. sweep     : carbon-price sweep (cost vs CO2 Pareto) on 2019 origins
6. validate  : planned inter-PADD flows vs EIA recorded movements
7. montecarlo: robustness of first-stage plans under sampled demand/outages
"""
from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from optigreen.data import eia
from optigreen.data import realcase as rc
from optigreen.forecasting import demand_model as dm
from optigreen.optimization import network_milp as nm
from optigreen.risk import outage_model as om

warnings.filterwarnings("ignore")

STRATEGIES = {
    "BAU rule": dict(kind="bau"),
    "MILP P50": dict(kind="milp", demand="P50", risk=None, carbon=0.0),
    "MILP P90": dict(kind="milp", demand="P90", risk=None, carbon=0.0),
    "MILP P50 + XGB risk": dict(kind="milp", demand="P50", risk="xgboost", carbon=0.0),
    "MILP P50 + GAT risk": dict(kind="milp", demand="P50", risk="gat", carbon=0.0),
    "MILP P50 + CO2 $100/t": dict(kind="milp", demand="P50", risk=None, carbon=100.0),
    "Perfect information": dict(kind="oracle", carbon=0.0),
}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- #
# 1. data
# --------------------------------------------------------------------------- #
def load_data(bulk: Optional[str], table_dir: str) -> Dict:
    if bulk:
        log(f"extracting EIA tables from {bulk}")
        eia.build_from_bulk(bulk, table_dir)
    t = eia.load_tables(table_dir)
    panel = rc.demand_panel(t)
    labels = rc.outage_labels(t)
    return {"tables": t, "panel": panel, "labels": labels}


def data_summary(d: Dict) -> Dict:
    t, panel, lab = d["tables"], d["panel"], d["labels"]
    su = t["series_used"]
    dem19 = panel[panel["month"].dt.year == 2019].groupby("product")["kt"].sum() / 1000.0
    return {
        "source": "U.S. EIA Petroleum bulk file PET.zip (public domain), https://www.eia.gov/opendata/bulk/PET.zip",
        "series_used": int(len(su)), "series_by_table": su.groupby("table").size().to_dict(),
        "demand_months": [str(panel["month"].min().date()), str(panel["month"].max().date())],
        "regions": int(panel["region"].nunique()), "states": int(panel[panel["kind"] == "state"]["region"].nunique()),
        "series_region_product": int(panel.groupby(["region", "product"]).ngroups),
        "imputed_share": float(panel["imputed"].mean()),
        "demand_2019_Mt": {k: round(v, 1) for k, v in dem19.items()},
        "districts": int(lab["district"].nunique()),
        "district_months": [str(lab["month"].min().date()), str(lab["month"].max().date())],
        "outage_rate": float(lab["outage"].mean()),
        "outage_events": int(lab["outage"].sum()),
    }


# --------------------------------------------------------------------------- #
# 2. forecasting
# --------------------------------------------------------------------------- #
def fit_forecasters(panel: pd.DataFrame, years: List[int]) -> Dict[int, dm.QuantileDemandForecaster]:
    out = {}
    for y in years:
        log(f"forecaster: fitting on targets <= {y - 1}-12")
        out[y] = dm.QuantileDemandForecaster().fit(panel, pd.Timestamp(f"{y - 1}-12-01"))
    return out


def forecast_eval(panel: pd.DataFrame, forecasters: Dict[int, dm.QuantileDemandForecaster],
                  origins: List[pd.Timestamp]) -> Dict:
    preds, bases = [], []
    for y, f in forecasters.items():
        oy = [o for o in origins if o.year == y]
        if not oy:
            continue
        rows = dm.build_rows(panel, oy)
        rows = rows[rows["target_month"] <= rc.DEMAND_END]
        preds.append(f.predict_rows(rows))
        bases.append(dm.baseline_predictions(rows))
    P, B = pd.concat(preds), pd.concat(bases)
    P = P.merge(B[["region", "product_name", "origin", "h", "snaive", "snaive_growth"]],
                on=["region", "product_name", "origin", "h"])
    res = {}
    periods = {"2018-2019 (pre-COVID)": (2018, 2019), "2020-2021 (COVID & recovery)": (2020, 2021)}
    for name, (a, b) in periods.items():
        sub = P[(P["origin"].dt.year >= a) & (P["origin"].dt.year <= b)]
        r = {"XGB quantile (P50)": dm.forecast_metrics(sub, "P50"),
             "Seasonal naive": dm.forecast_metrics(sub, "snaive", False),
             "Seasonal naive + growth": dm.forecast_metrics(sub, "snaive_growth", False)}
        r["by_product"] = {k: {"XGB": dm.forecast_metrics(g, "P50")["WAPE"],
                               "SNaive+growth": dm.forecast_metrics(g, "snaive_growth", False)["WAPE"],
                               "coverage_80": dm.forecast_metrics(g, "P50")["coverage_80"]}
                           for k, g in sub.groupby("product_name")}
        r["by_horizon"] = {int(h): dm.forecast_metrics(g, "P50") for h, g in sub.groupby("h")}
        r["wape_gain_vs_snaive_growth_CI"] = _wape_diff_ci(sub)
        res[name] = r
    return {"metrics": res, "predictions": P}


def _wape_diff_ci(df: pd.DataFrame, n_boot: int = 400, seed: int = 1) -> List[float]:
    """Block bootstrap (by origin month) for the relative WAPE reduction vs seasonal naive + growth."""
    rng = np.random.default_rng(seed)
    g = {o: x for o, x in df.groupby("origin")}
    keys = list(g)
    vals = []
    for _ in range(n_boot):
        s = pd.concat([g[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        a = np.abs(s["y"] - s["P50"]).sum()
        b = np.abs(s["y"] - s["snaive_growth"]).sum()
        vals.append(1 - a / b)
    return [float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# --------------------------------------------------------------------------- #
# 3. risk
# --------------------------------------------------------------------------- #
RISK_SPLIT = {"train_end": "2012-10-01", "val": ("2013-01-01", "2016-10-01"), "test_start": "2017-01-01"}


def risk_rows(labels: pd.DataFrame) -> pd.DataFrame:
    end = labels["month"].max() - pd.DateOffset(months=2)
    return om.build_rows(labels, pd.date_range("1988-01-01", end, freq="MS")).dropna(subset=["y"])


def risk_offline(rows: pd.DataFrame) -> Dict:
    tr = rows[rows["origin"] <= RISK_SPLIT["train_end"]]
    va = rows[(rows["origin"] >= RISK_SPLIT["val"][0]) & (rows["origin"] <= RISK_SPLIT["val"][1])]
    te = rows[rows["origin"] >= RISK_SPLIT["test_start"]]
    models = {"logistic": om.TabularRisk("logistic").fit(tr, va), "xgboost": om.TabularRisk("xgboost").fit(tr, va),
              "gat": om.GATRisk().fit(tr, va), "gat_self": om.GATRisk(self_only=True).fit(tr, va)}
    df = te[["origin", "target_month", "district", "h", "y", "severity"]].copy()
    df["climatology"] = om.climatology(te)
    for k, m in models.items():
        df[k] = m.predict(te)
    cols = ["climatology", "logistic", "xgboost", "gat", "gat_self"]
    met = {c: om.risk_metrics(df["y"].to_numpy(), df[c].to_numpy(), df["climatology"].to_numpy()) for c in cols}
    ci = om.block_bootstrap_ci(df, cols, "AP", 400)
    ci_auc = om.block_bootstrap_ci(df, cols, "ROC_AUC", 400)
    att = models["gat"].attention(te[te["h"] == 1])
    split = {"train": [str(tr["origin"].min().date()), str(tr["origin"].max().date()), int(len(tr)), float(tr["y"].mean())],
             "val": [str(va["origin"].min().date()), str(va["origin"].max().date()), int(len(va)), float(va["y"].mean())],
             "test": [str(te["origin"].min().date()), str(te["origin"].max().date()), int(len(te)), float(te["y"].mean())]}
    return {"metrics": met, "ap_ci": ci, "auc_ci": ci_auc, "predictions": df, "attention": att,
            "split": split, "gat_epochs": [models["gat"].best_epoch, models["gat_self"].best_epoch]}


def graph_checks(labels: pd.DataFrame) -> Dict:
    """Do graph edges carry information? Co-outage lift on edges vs non-edges."""
    A = om.district_graph()
    D = om.DISTRICTS
    piv = labels.pivot_table(index="month", columns="district", values="outage").dropna()
    piv = piv[D]
    p = piv.mean()
    lifts_e, lifts_n = [], []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            both = (piv[D[i]] * piv[D[j]]).mean()
            exp = p[D[i]] * p[D[j]]
            lift = both / exp if exp > 0 else np.nan
            (lifts_e if A[i, j] > 0 else lifts_n).append(lift)
    lag = []  # next-month lift: outage in j at t+1 given outage in neighbour i at t
    for i in range(len(D)):
        for j in range(len(D)):
            if i == j or A[i, j] == 0:
                continue
            a = piv[D[i]].iloc[:-1].to_numpy()
            b = piv[D[j]].iloc[1:].to_numpy()
            if a.sum() > 0 and b.mean() > 0:
                lag.append(b[a == 1].mean() / b.mean())
    return {"nodes": len(D), "edges_undirected": int((A.sum() - len(D)) / 2),
            "co_outage_lift_edges_mean": float(np.nanmean(lifts_e)),
            "co_outage_lift_nonedges_mean": float(np.nanmean(lifts_n)),
            "next_month_lift_neighbours_mean": float(np.nanmean(lag)) if lag else None,
            "months_used": int(len(piv))}


def fit_risk_for_year(rows: pd.DataFrame, year: int) -> Dict:
    tr = rows[rows["origin"] <= f"{year - 5}-10-01"]
    va = rows[(rows["origin"] >= f"{year - 4}-01-01") & (rows["origin"] <= f"{year - 1}-10-01")]
    return {"xgboost": om.TabularRisk("xgboost").fit(tr, va), "gat": om.GATRisk().fit(tr, va)}


def risk_for_origin(rows_all: pd.DataFrame, models: Dict, origin: pd.Timestamp, labels: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    r = om.build_rows(labels, [origin])
    out = {}
    for k, m in models.items():
        p = m.predict(r)
        out[k] = pd.DataFrame({"district": r["district"], "target_month": r["target_month"], "p": p})
    return out


# --------------------------------------------------------------------------- #
# 4. backtest
# --------------------------------------------------------------------------- #
def run_origin(d: Dict, origin: pd.Timestamp, forecaster: dm.QuantileDemandForecaster,
               risk_models: Dict, strategies: Dict = STRATEGIES, keep_flows: bool = True) -> Dict:
    t, panel, lab = d["tables"], d["panel"], d["labels"]
    fc = forecaster.forecast(panel, origin)
    risk = risk_for_origin(None, risk_models, origin, lab)
    inp = {q: nm.build_inputs(t, panel, fc, origin, demand_col=q, labels=lab) for q in ("P50", "P90")}
    inp_act = nm.build_inputs(t, panel, None, origin, labels=lab)
    avail = nm.actual_availability(inp_act, lab)
    out = {"origin": origin, "rows": [], "flows": None, "plans": {}}
    for name, s in strategies.items():
        t0 = time.time()
        if s["kind"] == "bau":
            fs = nm.bau_first_stage(inp["P50"], t)
            planned = None
        elif s["kind"] == "oracle":
            o_inp = nm.build_inputs(t, panel, None, origin, labels=lab)
            m = nm.build_model(o_inp, availability=avail)
            st, el, gap = nm.solve(m, 60, 0.002)
            planned = nm.extract(m, o_inp, st, el, gap)
            fs = nm.first_stage(m)
        else:
            pi = inp[s["demand"]]
            if s["risk"]:
                pi = nm.build_inputs(t, panel, fc, origin, demand_col=s["demand"], labels=lab, risk=risk[s["risk"]])
            m = nm.build_model(pi, carbon_kusd_per_kt=s["carbon"], risk_aware=bool(s["risk"]))
            st, el, gap = nm.solve(m, 60, 0.002)
            planned = nm.extract(m, pi, st, el, gap)
            fs = nm.first_stage(m)
        ev = nm.evaluate(inp_act, fs, avail, carbon_kusd_per_kt=s.get("carbon", 0.0))
        row = {"origin": origin, "strategy": name, "eval_status": ev.status, "seconds": time.time() - t0}
        row.update({f"real_{k}": v for k, v in ev.kpis.items()})
        if planned is not None:
            row.update({f"plan_{k}": v for k, v in planned.kpis.items()})
            row["plan_status"] = planned.status
            row["plan_gap"] = planned.mip_gap
            row["plan_n_int"] = planned.n_int
            row["plan_n_vars"] = planned.n_vars
            row["plan_n_cons"] = planned.n_cons
            row["plan_solve_s"] = planned.solve_s
            if keep_flows and name == "MILP P50":
                out["flows"] = planned.flows_dh.assign(origin=origin)
                out["crude"] = planned.crude.assign(origin=origin)
                out["plans"][name] = planned
        out["rows"].append(row)
    return out


def run_backtest(d: Dict, origins: List[pd.Timestamp], forecasters: Dict, risk_by_year: Dict,
                 out_dir: str) -> Dict:
    rows, flows, crude = [], [], []
    for o in origins:
        t0 = time.time()
        r = run_origin(d, o, forecasters[o.year], risk_by_year[o.year])
        rows += r["rows"]
        if r["flows"] is not None:
            flows.append(r["flows"])
            crude.append(r["crude"])
        log(f"backtest {o.date()} done in {time.time() - t0:.0f}s")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "backtest_by_origin.csv"), index=False)
    df = pd.DataFrame(rows)
    fl = pd.concat(flows) if flows else pd.DataFrame()
    cr = pd.concat(crude) if crude else pd.DataFrame()
    return {"by_origin": df, "flows": fl, "crude": cr}


def summarise_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = np.where(df["origin"].dt.year <= 2019, "2018-2019", "2020-2021")
    agg = {"real_fill_rate": "mean", "real_unmet_kt": "sum", "real_cost_total_musd": "sum",
           "real_cost_supply_chain_musd": "sum", "real_cost_logistics_musd": "sum", "real_co2_total_kt": "sum",
           "real_emergency_imports_kt": "sum", "real_export_short_kt": "sum", "real_demand_kt": "sum"}
    s = df.groupby(["period", "strategy"]).agg(agg).reset_index()
    a = df.groupby("strategy").agg(agg).reset_index().assign(period="all")
    s = pd.concat([s, a])
    s["real_fill_volume"] = 1 - s["real_unmet_kt"] / s["real_demand_kt"]
    return s


def paired_ci(df: pd.DataFrame, a: str, b: str, col: str, n_boot: int = 2000, seed: int = 3) -> List[float]:
    """Bootstrap over origins of mean(col[a] - col[b])."""
    x = df[df["strategy"] == a].set_index("origin")[col]
    y = df[df["strategy"] == b].set_index("origin")[col]
    dlt = (x - y).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(n_boot)]
    return [float(dlt.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# --------------------------------------------------------------------------- #
# 5. carbon sweep
# --------------------------------------------------------------------------- #
def carbon_sweep(d: Dict, origins: List[pd.Timestamp], forecaster: dm.QuantileDemandForecaster,
                 lambdas=(0, 25, 50, 100, 200, 400)) -> pd.DataFrame:
    t, panel, lab = d["tables"], d["panel"], d["labels"]
    rows = []
    for o in origins:
        fc = forecaster.forecast(panel, o)
        inp = nm.build_inputs(t, panel, fc, o, labels=lab)
        for lam in lambdas:
            res = nm.plan(inp, carbon_kusd_per_kt=float(lam))
            k = res.kpis
            rows.append({"origin": o, "lambda_usd_per_t": lam, "cost_supply_chain_musd": k["cost_supply_chain_musd"],
                         "cost_total_musd": k["cost_total_musd"], "co2_total_kt": k["co2_total_kt"],
                         "co2_refining_kt": k["co2_refining_kt"], "co2_transport_kt": k["co2_transport_kt"],
                         "co2_imports_kt": k["co2_imports_kt"], "fill_rate": k["fill_rate"],
                         "crude_kbbl": k["crude_kbbl"], "imports_kt": k["imports_kt"],
                         "surplus_exports_kt": k["surplus_exports_kt"], "co2_intensity": k["co2_intensity_kg_per_t"],
                         "logistics_musd": k["cost_logistics_musd"]})
        log(f"carbon sweep {o.date()} done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. validation of planned flows vs reality
# --------------------------------------------------------------------------- #
def flow_validation(d: Dict, flows: pd.DataFrame, crude: pd.DataFrame) -> Dict:
    t, lab = d["tables"], d["labels"]
    mv = rc.movements_panel(t)
    mv["lane"] = mv["from_area"] + ">" + mv["to_area"] + ":" + mv["mode"]
    act = mv.groupby(["month", "lane"])["kt"].sum()
    # compare the first month of every plan (each calendar month planned exactly once)
    first_f = flows[(flows["lane"].notna()) & (flows["month"] == flows["origin"])]
    pl = first_f.groupby(["month", "lane"])["kt"].sum()
    j = pd.concat([pl.rename("plan"), act.rename("actual")], axis=1).dropna(subset=["plan"]).fillna(0.0)
    j = j[j.index.get_level_values(0).isin(pl.index.get_level_values(0))]
    by_lane = j.groupby(level=1).sum()
    c_util = (crude[crude["month"] == crude["origin"]].groupby(["month", "district"])["utilization"]
              .mean().rename("plan").reset_index())
    c_util = c_util.merge(lab[["month", "district", "utilization_pct"]], on=["month", "district"])
    c_util["actual"] = c_util["utilization_pct"] / 100
    first = crude[crude["month"] == crude["origin"]]
    sys_plan = first.groupby("month")["crude_kbbl"].sum()
    dp = rc.district_panel(t).groupby("month")["crude_kbbl"].sum()
    sys = pd.concat([sys_plan.rename("plan"), dp.rename("actual")], axis=1).dropna()
    pre = j[j.index.get_level_values(0) < pd.Timestamp("2020-01-01")]
    sys_pre = sys[sys.index < pd.Timestamp("2020-01-01")]
    util_pre = c_util[c_util["month"] < pd.Timestamp("2020-01-01")]
    return {"lane_corr_pearson": float(j.corr().iloc[0, 1]), "lane_corr_spearman": float(j.corr("spearman").iloc[0, 1]),
            "pre2020_lane_corr_pearson": float(pre.corr().iloc[0, 1]),
            "pre2020_lane_corr_spearman": float(pre.corr("spearman").iloc[0, 1]),
            "pre2020_system_crude_plan_vs_actual_pct": float((sys_pre["plan"].sum() / sys_pre["actual"].sum() - 1) * 100),
            "pre2020_util_mae": float((util_pre["plan"] - util_pre["actual"]).abs().mean()),
            "lane_share_direction_agree": float(((j["plan"] > 0) == (j["actual"] > 0)).mean()),
            "by_lane": by_lane, "pairs": j.reset_index(),
            "util_mae": float((c_util["plan"] - c_util["actual"]).abs().mean()),
            "system_crude_plan_vs_actual_pct": float((sys["plan"].sum() / sys["actual"].sum() - 1) * 100),
            "util_pairs": c_util}


# --------------------------------------------------------------------------- #
# 7. Monte Carlo robustness (Stage-3 preview)
# --------------------------------------------------------------------------- #
def sample_demand(fc: pd.DataFrame, rng: np.random.Generator, rho: float = 0.5) -> pd.DataFrame:
    """Piecewise-linear quantile sampling with a common factor (correlation rho)."""
    from scipy.stats import norm
    z_c = rng.standard_normal()
    n = len(fc)
    z = np.sqrt(rho) * z_c + np.sqrt(1 - rho) * rng.standard_normal(n)
    u = norm.cdf(z)
    p10, p50, p90 = fc["P10"].to_numpy(), fc["P50"].to_numpy(), fc["P90"].to_numpy()
    v = np.where(u < 0.5, p50 - (0.5 - u) / 0.4 * (p50 - p10), p50 + (u - 0.5) / 0.4 * (p90 - p50))
    out = fc.copy()
    out["sample"] = np.clip(v, 0, None)
    return out


def monte_carlo(d: Dict, origin: pd.Timestamp, forecaster, risk_models, n: int = 30, seed: int = 7,
                strategies=("BAU rule", "MILP P50", "MILP P90", "MILP P50 + XGB risk")) -> pd.DataFrame:
    t, panel, lab = d["tables"], d["panel"], d["labels"]
    rng = np.random.default_rng(seed)
    fc = forecaster.forecast(panel, origin)
    risk = risk_for_origin(None, risk_models, origin, lab)
    base = nm.build_inputs(t, panel, fc, origin, labels=lab)
    fs = {}
    for s in strategies:
        cfg = STRATEGIES[s]
        if cfg["kind"] == "bau":
            fs[s] = nm.bau_first_stage(base, t)
            continue
        pi = nm.build_inputs(t, panel, fc, origin, demand_col=cfg["demand"], labels=lab,
                             risk=risk[cfg["risk"]] if cfg["risk"] else None)
        m = nm.build_model(pi, risk_aware=bool(cfg["risk"]))
        nm.solve(m, 60, 0.002)
        fs[s] = nm.first_stage(m)
    past = lab[(lab["month"] < origin) & (lab["outage"] == 1)]
    sev_pool = past["severity"].to_numpy()
    rx = risk["xgboost"]
    rows = []
    for i in range(n):
        smp = sample_demand(fc, rng)
        smp = smp.rename(columns={"sample": "SAMPLE"})
        inp_s = nm.build_inputs(t, panel, smp, origin, demand_col="SAMPLE", labels=lab)
        avail = {}
        for dd in inp_s.districts.index:
            for tt, mth in enumerate(inp_s.months):
                cap = inp_s.districts.loc[dd, "cap_kbcd"] * mth.days_in_month * inp_s.districts.loc[dd, "umax"]
                p = rx[(rx["district"] == dd) & (rx["target_month"] == mth)]["p"]
                p = float(p.iloc[0]) if len(p) else 0.05
                avail[(dd, tt)] = cap * (1 - rng.choice(sev_pool)) if rng.random() < p else cap
        for s in strategies:
            ev = nm.evaluate(inp_s, fs[s], avail)
            rows.append({"sample": i, "strategy": s, "fill_rate": ev.kpis["fill_rate"],
                         "unmet_kt": ev.kpis["unmet_kt"], "cost_total_musd": ev.kpis["cost_total_musd"],
                         "cost_supply_chain_musd": ev.kpis["cost_supply_chain_musd"],
                         "co2_total_kt": ev.kpis["co2_total_kt"],
                         "emergency_imports_kt": ev.kpis["emergency_imports_kt"]})
        log(f"monte carlo sample {i + 1}/{n}")
    return pd.DataFrame(rows)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o
