"""
OptiGreen-Chem dashboard (Stage 2, real EIA data).

Run:  python -m streamlit run app/streamlit_app.py

Reads the outputs of `scripts/run_stage2.py` (results/stage2/) and solves the
planning MILP live for any month between Jan 2018 and Jan 2022.
Every number on these screens is computed from data; nothing is a placeholder.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from optigreen.data import eia, network as net, realcase as rc  # noqa: E402
from optigreen.optimization import network_milp as nm  # noqa: E402

RES = ROOT / "results" / "stage2"
TABLES = ROOT / "data" / "real" / "eia"
st.set_page_config(page_title="OptiGreen-Chem", page_icon="🌿", layout="wide")
PRODUCT_LABEL = {k: v for k, v in eia.PRODUCT_NAMES.items()}


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading EIA tables …")
def load_core():
    t = eia.load_tables(str(TABLES))
    panel = rc.demand_panel(t)
    labels = rc.outage_labels(t)
    return t, panel, labels


@st.cache_resource(show_spinner="Loading trained models …")
def load_models():
    import gzip
    p = RES / "models.pkl.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_csv(name):
    p = RES / name
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ("origin", "target_month", "month"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data
def load_summary():
    p = RES / "summary.json"
    return json.load(open(p)) if p.exists() else {}


def missing_results():
    st.warning("Stage-2 results not found. Run `python scripts/run_stage2.py --bulk PET.zip` first "
               "(or `--quick` for a 5-minute smoke test).")
    st.stop()


t, panel, labels = load_core()
summary = load_summary()

st.sidebar.title("🌿 OptiGreen-Chem")
st.sidebar.caption("Stage 2 · real U.S. EIA refined-products data")
tab = st.sidebar.radio("Navigation", ["Overview", "Demand forecast", "Risk intelligence", "Optimization",
                                      "Sustainability", "Robustness & backtest", "Validation"])


# --------------------------------------------------------------------------- #
def kpi_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


if tab == "Overview":
    st.title("OptiGreen-Chem: probabilistic demand → risk → MILP → cost/CO₂")
    ds = summary.get("data", {})
    kpi_row([("Refining districts (plants)", len(net.DISTRICT_INFO)), ("Terminal hubs", len(net.HUBS)),
             ("Demand regions", panel["region"].nunique()), ("Products", len(eia.PRODUCTS)),
             ("EIA series used", ds.get("series_used", "–"))])
    st.markdown("""
**Data.** U.S. Energy Information Administration petroleum bulk file (PET.zip, public domain): state-level
prime-supplier sales 1993-2022 (demand), refining-district capacity, utilisation, crude runs and product output
1985-2026 (plants and outages), sub-PADD stocks (warehouses), inter-PADD movements by pipeline/tanker/barge
(routes), imports/exports, spot prices and refinery fuel use (CO₂).

**Pipeline.** (1) quantile XGBoost gives P10/P50/P90 demand for every state × product, 1-3 months ahead;
(2) outage models (XGBoost, graph attention network) give the probability that each refining district loses
throughput; (3) a Pyomo/HiGHS MILP plans crude runs, pipeline/tanker/barge shipments (integer cargoes),
imports, inventories and deliveries; (4) cost and CO₂ are traded off with a carbon price, and plans are
re-scored against what actually happened (actual demand, actual outages).
""")
    if ds:
        st.subheader("Dataset")
        st.json(ds)
    st.subheader("Network")
    nodes = ([{"type": "Refining district", "name": f"{k} {v[3]}", "lat": v[1], "lon": v[2]} for k, v in net.DISTRICT_INFO.items()]
             + [{"type": "Terminal hub", "name": f"{k} {v[4]}", "lat": v[1], "lon": v[2]} for k, v in net.HUBS.items()]
             + [{"type": "Demand state", "name": k, "lat": v[1], "lon": v[2]} for k, v in net.STATE_INFO.items()])
    fig = px.scatter_geo(pd.DataFrame(nodes), lat="lat", lon="lon", color="type", hover_name="name", scope="usa",
                         color_discrete_sequence=["#c2185b", "#1565c0", "#9e9e9e"])
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")

elif tab == "Demand forecast":
    st.title("Probabilistic demand forecast (P10 / P50 / P90)")
    fp = load_csv("forecast_predictions.csv.gz")
    if fp is None:
        missing_results()
    c1, c2, c3 = st.columns(3)
    prod = c1.selectbox("Product", eia.PRODUCTS, format_func=lambda k: PRODUCT_LABEL[k])
    regs = sorted(fp[fp["product_name"] == prod]["region"].unique())
    reg = c2.selectbox("Region", regs, index=regs.index("TX") if "TX" in regs else 0)
    h = c3.selectbox("Horizon (months ahead)", [1, 2, 3])
    sub = fp[(fp["region"] == reg) & (fp["product_name"] == prod) & (fp["h"] == h)].sort_values("target_month")
    hist = panel[(panel["region"] == reg) & (panel["product"] == prod) & (panel["month"] >= "2015-01-01")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(sub["target_month"]) + list(sub["target_month"])[::-1],
                             y=list(sub["P90"]) + list(sub["P10"])[::-1], fill="toself",
                             fillcolor="rgba(21,101,192,0.18)", line=dict(width=0), name="P10–P90"))
    fig.add_trace(go.Scatter(x=sub["target_month"], y=sub["P50"], name="P50", line=dict(color="#1565c0", dash="dash")))
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["kt"], name="Actual", line=dict(color="#222")))
    fig.add_trace(go.Scatter(x=sub["target_month"], y=sub["snaive_growth"], name="Seasonal naive + growth",
                             line=dict(color="#999", dash="dot")))
    fig.update_layout(height=460, yaxis_title="kt per month", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Forecasts are out-of-sample: each year's model is fit only on data before that year; the forecast for "
               "month t made at origin o uses data up to o−1.")
    fm = summary.get("forecast", {})
    for period, r in fm.items():
        st.subheader(period)
        tab_rows = [{"Model": k, "WAPE": v["WAPE"], "Bias %": v["bias_pct"], "P10–P90 coverage": v.get("coverage_80")}
                    for k, v in r.items() if isinstance(v, dict) and "WAPE" in v]
        st.dataframe(pd.DataFrame(tab_rows).style.format({"WAPE": "{:.2%}", "Bias %": "{:.2f}",
                                                          "P10–P90 coverage": "{:.1%}"}, na_rep="–"))

elif tab == "Risk intelligence":
    st.title("Refinery-district outage risk")
    rp = load_csv("risk_test_predictions.csv")
    if rp is None:
        missing_results()
    rm = summary.get("risk", {})
    if rm:
        mt = pd.DataFrame(rm["metrics"]).T[["ROC_AUC", "AP", "Brier", "BrierSkill_vs_clim"]]
        ci = rm.get("ap_ci", {})
        mt["AP 95% CI"] = [f"{ci[k][1]:.3f}–{ci[k][2]:.3f}" if k in ci else "" for k in mt.index]
        st.dataframe(mt.style.format({"ROC_AUC": "{:.3f}", "AP": "{:.3f}", "Brier": "{:.4f}",
                                      "BrierSkill_vs_clim": "{:+.3f}"}))
        st.caption(f"Test origins {rm['split']['test'][0]} → {rm['split']['test'][1]}; outage base rate "
                   f"{rm['split']['test'][3]:.1%}. gat_self = same GAT with self-loops only (no message passing).")
    model = st.selectbox("Model", ["xgboost", "gat", "climatology", "gat_self", "logistic"])
    h = st.selectbox("Horizon", [1, 2, 3])
    sub = rp[rp["h"] == h]
    pv = sub.pivot_table(index="district", columns="target_month", values=model)
    fig = px.imshow(pv, aspect="auto", color_continuous_scale="Reds", labels=dict(color="P(outage)"))
    ev = sub[sub["y"] == 1]
    fig.add_trace(go.Scatter(x=ev["target_month"], y=ev["district"], mode="markers", name="actual outage",
                             marker=dict(symbol="x", color="black", size=6)))
    fig.update_layout(height=430, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    g = rm.get("graph", {})
    if g:
        st.subheader("Does the district graph carry information?")
        st.write(f"Edges: {g['edges_undirected']}. Co-outage lift on connected pairs "
                 f"{g['co_outage_lift_edges_mean']:.2f}× vs {g['co_outage_lift_nonedges_mean']:.2f}× on unconnected pairs; "
                 f"next-month lift after a neighbour's outage {g['next_month_lift_neighbours_mean']:.2f}×.")
    att = RES / "gat_attention.csv"
    if att.exists():
        st.plotly_chart(px.imshow(pd.read_csv(att, index_col=0), color_continuous_scale="Blues",
                                  labels=dict(color="attention")), width="stretch")

elif tab == "Optimization":
    st.title("Plan a quarter (live MILP)")
    models = load_models()
    if models is None:
        missing_results()
    with st.form("plan"):
        c1, c2, c3, c4 = st.columns(4)
        months = pd.date_range("2018-01-01", "2022-01-01", freq="MS")
        origin = c1.selectbox("First planning month", months, index=list(months).index(pd.Timestamp("2019-08-01")),
                              format_func=lambda d: d.strftime("%b %Y"))
        quant = c2.selectbox("Demand forecast", ["P50", "P90", "P10"])
        risk_on = c3.selectbox("Risk", ["None", "XGBoost outage risk"])
        carbon = c4.slider("Carbon price ($/t CO₂)", 0, 400, 0, 25)
        go_btn = st.form_submit_button("Solve", type="primary")
    if go_btn:
        with st.spinner("Building inputs from EIA data and solving with HiGHS …"):
            y = min(max(origin.year, 2018), max(models["forecasters"]))
            fc = models["forecasters"][y].forecast(panel, origin)
            risk = None
            if risk_on != "None":
                from optigreen.risk import outage_model as om
                r = om.build_rows(labels, [origin])
                p = models["risk_by_year"][y]["xgboost"].predict(r)
                risk = pd.DataFrame({"district": r["district"], "target_month": r["target_month"], "p": p})
            inp = nm.build_inputs(t, panel, fc, origin, demand_col=quant, labels=labels, risk=risk)
            t0 = time.time()
            res = nm.plan(inp, carbon_kusd_per_kt=float(carbon), risk_aware=risk is not None)
            st.session_state["plan_result"] = (res, inp, time.time() - t0)
    if "plan_result" in st.session_state:
        res, inp, wall = st.session_state["plan_result"]
        k = res.kpis
        kpi_row([("Supply-chain cost", f"${k['cost_supply_chain_musd'] / 1000:,.2f} bn"),
                 ("Planned fill rate", f"{k['fill_rate']:.2%}"), ("CO₂", f"{k['co2_total_kt'] / 1000:,.2f} Mt"),
                 ("Solver time", f"{res.solve_s:.2f} s"), ("Status", res.status)])
        st.caption(f"{res.n_vars:,} variables ({res.n_int} integer: tanker/barge/import cargoes), {res.n_cons:,} constraints; "
                   f"MIP gap {res.mip_gap if res.mip_gap is not None else 0:.4%}. Horizon "
                   f"{inp.months[0]:%b %Y} – {inp.months[-1]:%b %Y}.")
        with st.expander("Cost and CO₂ breakdown", expanded=True):
            cb = pd.DataFrame([{"item": c.replace("cost_", "").replace("_musd", ""), "USD million": v}
                               for c, v in k.items() if c.startswith("cost_") and c.endswith("_musd")
                               and c not in ("cost_total_musd", "cost_supply_chain_musd", "cost_logistics_musd")])
            cc = pd.DataFrame([{"source": c.replace("co2_", "").replace("_kt", ""), "kt CO2": v}
                               for c, v in k.items() if c.startswith("co2_") and c.endswith("_kt") and c != "co2_total_kt"])
            c1, c2 = st.columns(2)
            c1.dataframe(cb.style.format({"USD million": "{:,.1f}"}))
            c2.dataframe(cc.style.format({"kt CO2": "{:,.1f}"}))
        st.subheader("Routing (month 1)")
        m0 = inp.months[0]
        f = res.flows_dh[res.flows_dh["month"] == m0].groupby(["district", "hub", "mode"])["kt"].sum().reset_index()
        q = res.flows_hr[res.flows_hr["month"] == m0].groupby(["hub", "region"])["kt"].sum().reset_index()
        fig = go.Figure()
        for r in q.itertuples(index=False):
            if r.region.startswith("POOL") or r.region not in net.STATE_INFO:
                continue
            a, b = net.hub_coord(r.hub), net.state_coord(r.region)
            fig.add_trace(go.Scattergeo(lat=[a[0], b[0]], lon=[a[1], b[1]], mode="lines", showlegend=False,
                                        line=dict(width=0.5 + 3 * r.kt / q["kt"].max(), color="rgba(21,101,192,0.45)")))
        for r in f.itertuples(index=False):
            a, b = net.district_coord(r.district), net.hub_coord(r.hub)
            fig.add_trace(go.Scattergeo(lat=[a[0], b[0]], lon=[a[1], b[1]], mode="lines", showlegend=False,
                                        line=dict(width=0.5 + 5 * r.kt / f["kt"].max(),
                                                  color="rgba(194,24,91,0.7)" if "marine" not in r.mode else "rgba(0,137,123,0.8)")))
        fig.add_trace(go.Scattergeo(lat=[v[1] for v in net.DISTRICT_INFO.values()], lon=[v[2] for v in net.DISTRICT_INFO.values()],
                                    text=list(net.DISTRICT_INFO), mode="markers+text", name="Refining districts",
                                    marker=dict(size=9, color="#c2185b")))
        fig.add_trace(go.Scattergeo(lat=[v[1] for v in net.HUBS.values()], lon=[v[2] for v in net.HUBS.values()],
                                    text=list(net.HUBS), mode="markers+text", name="Hubs", marker=dict(size=9, color="#1565c0")))
        fig.update_layout(geo=dict(scope="usa"), height=560, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Pink: refinery → hub by pipeline; teal: tanker/barge; blue: hub → state deliveries.")
        c1, c2 = st.columns(2)
        c1.subheader("Crude runs")
        c1.dataframe(res.crude.assign(month=res.crude["month"].dt.strftime("%Y-%m"))
                     .style.format({"crude_kbbl": "{:,.0f}", "cap_kbbl": "{:,.0f}", "utilization": "{:.1%}"}))
        c2.subheader("Marine and import cargoes")
        c2.dataframe(res.cargoes.assign(month=res.cargoes["month"].dt.strftime("%Y-%m")))
        st.subheader("Shipment schedule (refinery → hub)")
        st.dataframe(res.flows_dh.assign(month=res.flows_dh["month"].dt.strftime("%Y-%m")).round(1))
        st.subheader("Shortages")
        sh = res.shortages[res.shortages["short_kt"] > 1e-3]
        st.dataframe(sh) if len(sh) else st.success("No planned shortage in any state/product/month.")

elif tab == "Sustainability":
    st.title("Cost vs CO₂ (carbon-price sweep)")
    sw = load_csv("carbon_sweep.csv")
    if sw is None:
        missing_results()
    agg = sw.groupby("lambda_usd_per_t")[["cost_total_musd", "co2_total_kt", "logistics_musd",
                                          "co2_refining_kt", "co2_transport_kt", "co2_imports_kt"]].mean().reset_index()
    base = agg.iloc[0]
    agg["Δcost (USD m)"] = agg["cost_total_musd"] - base["cost_total_musd"]
    agg["ΔCO2 %"] = (agg["co2_total_kt"] / base["co2_total_kt"] - 1) * 100
    fig = px.line(agg, x="co2_total_kt", y="cost_total_musd", text="lambda_usd_per_t", markers=True,
                  labels={"co2_total_kt": "CO₂ per quarter (kt)",
                          "cost_total_musd": "Cost excl. carbon charge (USD m)"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(agg.style.format("{:,.2f}"))
    st.caption("Mean over 2019 planning origins; each point is a full MILP solve. Labels: carbon price in $/t CO₂.")

elif tab == "Robustness & backtest":
    st.title("Backtest: plans re-scored against what actually happened")
    bs = load_csv("backtest_summary.csv")
    bo = load_csv("backtest_by_origin.csv")
    if bs is None:
        missing_results()
    per = st.selectbox("Period", ["all", "2018-2019", "2020-2021"])
    s = bs[bs["period"] == per].set_index("strategy")
    st.dataframe(s[["real_fill_volume", "real_unmet_kt", "real_cost_supply_chain_musd", "real_cost_total_musd",
                    "real_co2_total_kt", "real_emergency_imports_kt"]]
                 .style.format({"real_fill_volume": "{:.3%}", "real_unmet_kt": "{:,.0f}",
                                "real_cost_supply_chain_musd": "{:,.0f}", "real_cost_total_musd": "{:,.0f}",
                                "real_co2_total_kt": "{:,.0f}", "real_emergency_imports_kt": "{:,.0f}"}))
    st.caption("Each origin: plan 3 months with forecasts/risk known at the time, then fix crude runs (−10 %/+5 % recourse), "
               "chartered cargoes and contracted imports, and re-optimise dispatch against actual demand and actual "
               "refinery availability. Cost totals include penalty valuations for unmet demand and end-stock shortfall.")
    fig = px.line(bo, x="origin", y="real_cost_total_musd", color="strategy")
    st.plotly_chart(fig, width="stretch")
    mc = load_csv("monte_carlo_2019-08.csv")
    if mc is not None:
        st.subheader("Monte Carlo, Aug–Oct 2019 (sampled demand and outages)")
        st.plotly_chart(px.box(mc, x="strategy", y="cost_total_musd", points="all"), width="stretch")
        st.dataframe(mc.groupby("strategy")[["fill_rate", "unmet_kt", "cost_total_musd", "co2_total_kt"]].describe().T)

elif tab == "Validation":
    st.title("Does the optimiser behave like the real system?")
    v = summary.get("validation", {})
    if not v:
        missing_results()
    kpi_row([("Lane-flow correlation (Pearson)", f"{v['lane_corr_pearson']:.3f}"),
             ("Lane-flow correlation (Spearman)", f"{v['lane_corr_spearman']:.3f}"),
             ("District utilisation MAE", f"{v['util_mae']:.1%}"),
             ("System crude run, plan vs actual", f"{v['system_crude_plan_vs_actual_pct']:+.1f}%")])
    if "pre2020_lane_corr_pearson" in v:
        st.caption("Pre-COVID origins only (2018–2019):")
        kpi_row([("Lane-flow correlation (Pearson)", f"{v['pre2020_lane_corr_pearson']:.3f}"),
                 ("Lane-flow correlation (Spearman)", f"{v['pre2020_lane_corr_spearman']:.3f}"),
                 ("District utilisation MAE", f"{v['pre2020_util_mae']:.1%}"),
                 ("System crude run, plan vs actual", f"{v['pre2020_system_crude_plan_vs_actual_pct']:+.1f}%")])
    pairs = load_csv("validation_lane_month_pairs.csv")
    fig = px.scatter(pairs, x="actual", y="plan", color="lane", log_x=True, log_y=True,
                     labels={"actual": "EIA recorded movement (kt/month)", "plan": "MILP planned (kt/month)"})
    st.plotly_chart(fig, width="stretch")
    up = load_csv("validation_utilization.csv")
    st.plotly_chart(px.scatter(up, x="actual", y="plan", color="district",
                               labels={"actual": "actual utilisation", "plan": "planned utilisation"}),
                    width="stretch")
