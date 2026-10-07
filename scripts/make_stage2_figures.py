"""
Figures for the Stage-2 review, drawn only from results/stage2 and data/real/eia.

    python scripts/make_stage2_figures.py

Dark theme matching the review deck. Categorical colours are the validated
8-slot palette (dark steps), assigned in fixed order.
"""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from optigreen.data import eia, network as net, realcase as rc  # noqa: E402

RES = os.path.join(ROOT, "results", "stage2")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

BG, PANEL, GRID = "#10141a", "#161b22", "#2e3238"
INK, INK2, INK3 = "#ffffff", "#c3c2b7", "#8b8a83"
S = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = S

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2.0,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def load():
    s = json.load(open(os.path.join(RES, "summary.json")))
    t = eia.load_tables(os.path.join(ROOT, "data", "real", "eia"))
    return s, t


def draw_states(ax, gj):
    for f in gj["features"]:
        if f["properties"]["postal"] in ("AK", "HI"):
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor="#1b212b", edgecolor="#3a414d", lw=0.5))
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.25)
    ax.axis("off")


# --------------------------------------------------------------------------- #
def fig_network_map(s, t):
    gj = json.load(open(os.path.join(ROOT, "assets", "us_states_naturalearth_50m.geojson")))
    plan = pickle.load(open(os.path.join(RES, "headline_plan_2019-08.pkl"), "rb"))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    draw_states(ax, gj)
    m0 = plan.flows_hr["month"].min()
    q = plan.flows_hr[plan.flows_hr["month"] == m0].groupby(["hub", "region"])["kt"].sum().reset_index()
    q = q[q["region"].isin(net.STATE_INFO) & ~q["region"].isin(["AK", "HI"])]
    for r in q.itertuples(index=False):
        a, b = net.hub_coord(r.hub), net.state_coord(r.region)
        ax.plot([a[1], b[1]], [a[0], b[0]], color=BLUE, alpha=0.55, lw=0.4 + 2.2 * r.kt / q["kt"].max(), zorder=2)
    f = plan.flows_dh[plan.flows_dh["month"] == m0].groupby(["district", "hub", "mode"])["kt"].sum().reset_index()
    for r in f.itertuples(index=False):
        if r.hub in ("H5N",) and r.district == "P5":
            pass
        a, b = net.district_coord(r.district), net.hub_coord(r.hub)
        col = AQUA if "marine" in r.mode or "barge" in r.mode else MAGENTA
        ax.plot([a[1], b[1]], [a[0], b[0]], color=col, alpha=0.9, lw=0.6 + 4.5 * r.kt / f["kt"].max(), zorder=3)
    for k, v in net.DISTRICT_INFO.items():
        ax.scatter(v[2], v[1], s=70, color=MAGENTA, edgecolor=BG, lw=1.5, zorder=5)
        dx, dy, ha = {"EC": (0.1, -1.25, "center"), "3B": (0.4, -0.2, "left")}.get(k, (0.4, 0.35, "left"))
        ax.text(v[2] + dx, v[1] + dy, k, fontsize=8, color=INK2, zorder=6, ha=ha)
    for k, v in net.HUBS.items():
        ax.scatter(v[2], v[1], s=80, marker="s", color=BLUE, edgecolor=BG, lw=1.5, zorder=5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MAGENTA, marker="o", lw=2, label="Refinery region → hub (pipeline)"),
         Line2D([], [], color=AQUA, lw=2, label="Tanker / barge route"),
         Line2D([], [], color=BLUE, marker="s", lw=1, label="Hub → state delivery")]
    ax.legend(handles=h, loc="lower left", fontsize=9)
    save(fig, "network_plan_map.png")


def fig_forecast_examples(s, t):
    fp = pd.read_csv(os.path.join(RES, "forecast_predictions.csv.gz"), parse_dates=["origin", "target_month"])
    panel = rc.demand_panel(t)
    cases = [("TX", "GAS", "Texas · gasoline"), ("NY", "DIS", "New York · diesel & heating oil"),
             ("FL", "JET", "Florida · jet fuel")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (reg, prod, title) in zip(axes, cases):
        sub = fp[(fp["region"] == reg) & (fp["product_name"] == prod) & (fp["h"] == 1)].sort_values("target_month")
        hist = panel[(panel["region"] == reg) & (panel["product"] == prod) & (panel["month"] >= "2016-06-01")]
        ax.fill_between(sub["target_month"], sub["P10"] / 1000, sub["P90"] / 1000, color=BLUE, alpha=0.25,
                        lw=0, label="Low–high range (P10–P90)")
        ax.plot(sub["target_month"], sub["P50"] / 1000, color=BLUE, lw=1.6, label="Forecast, 1 month ahead (P50)")
        ax.plot(hist["month"], hist["kt"] / 1000, color=INK, lw=1.2, label="Actual")
        ax.axvline(pd.Timestamp("2020-03-01"), color=INK3, lw=0.8, ls=":")
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylabel("Million tonnes per month")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].text(pd.Timestamp("2020-04-01"), axes[1].get_ylim()[1] * 0.97, "COVID-19", color=INK3, fontsize=8, va="top")
    fig.tight_layout()
    save(fig, "forecast_examples.png")


def fig_forecast_by_product(s):
    r = s["forecast"]["2018-2019 (pre-COVID)"]["by_product"]
    prods = ["GAS", "DIS", "JET", "RES", "LPG"]
    lab = {"GAS": "Gasoline", "DIS": "Distillate", "JET": "Jet fuel", "RES": "Residual", "LPG": "Propane"}
    x = np.arange(len(prods))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    a = [r[p]["SNaive+growth"] * 100 for p in prods]
    b = [r[p]["XGB"] * 100 for p in prods]
    ax.bar(x - w / 2 - 0.01, a, w, color=INK3, label="Seasonal naive + growth")
    ax.bar(x + w / 2 + 0.01, b, w, color=BLUE, label="Quantile XGBoost (P50)")
    for i in range(len(prods)):
        ax.text(x[i] + w / 2 + 0.01, b[i] + 0.3, f"{b[i]:.1f}", ha="center", fontsize=9, color=INK2)
        ax.text(x[i] - w / 2 - 0.01, a[i] + 0.3, f"{a[i]:.1f}", ha="center", fontsize=9, color=INK3)
    ax.set_xticks(x, [lab[p] for p in prods])
    ax.set_ylabel("WAPE, % (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "forecast_wape_by_product.png")


def fig_risk_timeline(s):
    rp = pd.read_csv(os.path.join(RES, "risk_test_predictions.csv"), parse_dates=["origin", "target_month"])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for dist, col, lab in [("3B", BLUE, "Texas Gulf Coast"), ("3C", ORANGE, "Louisiana Gulf Coast"),
                           ("EC", AQUA, "East Coast")]:
        sub = rp[(rp["district"] == dist) & (rp["h"] == 1)].sort_values("target_month")
        ax.plot(sub["target_month"], sub["xgboost"], color=col, lw=1.6, label=lab)
        ev = sub[sub["y"] == 1]
        ax.scatter(ev["target_month"], ev["xgboost"], color=col, s=34, zorder=4, edgecolor=BG)
    top = ax.get_ylim()[1]
    for i, (d, txt) in enumerate([("2017-09-01", "Harvey"), ("2019-07-01", "Philadelphia fire"), ("2020-09-01", "Laura"),
                                  ("2021-02-01", "Texas freeze"), ("2021-09-01", "Ida")]):
        ax.axvline(pd.Timestamp(d), color=INK3, lw=0.7, ls=":")
        ax.text(pd.Timestamp(d), top * (1.0 if i % 2 == 0 else 0.92), " " + txt, fontsize=8.5, color=INK2, va="top")
    ax.set_ylim(0, top * 1.02)
    ax.set_ylabel("Chance of an outage")
    ax.legend(fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Predicted chance of an outage next month (XGBoost); dots = months that really were outages",
                 loc="left", fontsize=10, color=INK2, fontweight="normal")
    save(fig, "risk_timeline.png")


def fig_risk_metrics(s):
    ci = s["risk"]["ap_ci"]
    m = s["risk"]["metrics"]
    names = [("climatology", "Seasonal pattern only"), ("logistic", "Logistic regression"), ("xgboost", "XGBoost"),
             ("gat", "Graph network"), ("gat_self", "Graph network, no links")]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, (k, lab) in enumerate(names):
        mean, lo, hi = ci[k]
        col = {"climatology": INK3, "logistic": INK3, "xgboost": BLUE}.get(k, VIOLET)
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.scatter(m[k]["AP"], i, color=col, s=60, zorder=3, edgecolor=BG)
        ax.text(hi + 0.006, i, f"{m[k]['AP']:.3f}", va="center", fontsize=9, color=INK2)
    ax.axvline(s["risk"]["split"]["test"][3], color=RED, lw=1, ls="--")
    ax.text(s["risk"]["split"]["test"][3], -0.55, " random guess", color=RED, fontsize=8.5, va="bottom")
    ax.set_yticks(range(len(names)), [n[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("How well real outages are ranked, 2017–2026 (average precision, 95 % range)")
    ax.grid(axis="y", visible=False)
    save(fig, "risk_ap_ci.png")


def fig_validation(s):
    pairs = pd.read_csv(os.path.join(RES, "validation_lane_month_pairs.csv"))
    bl = pairs.groupby("lane")[["plan", "actual"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    lim = [10, max(bl["plan"].max(), bl["actual"].max()) * 1.6]
    ax.plot(lim, lim, color=INK3, lw=1, ls="--")
    bl = bl[(bl["plan"] > 0) | (bl["actual"] > 0)]
    ax.scatter(bl["actual"].clip(lower=12), bl["plan"].clip(lower=12), s=46, color=BLUE, edgecolor=BG, zorder=3)
    # hand-placed offsets keep the labels of neighbouring lanes apart
    names = {"P3>P1": "Gulf → East Coast pipeline", "P3>1C": "Gulf → Florida tanker",
             "P3>P5": "Gulf → West Coast pipeline", "P3>P2": "Gulf → Midwest pipeline",
             "P2>P4": "Midwest → Rockies pipeline", "P1>P2": "East Coast → Midwest pipeline"}
    offsets = {"P3>P1": (-8, 2, "right"), "P3>P5": (-6, 7, "right"), "P3>P2": (6, 5, "left"),
               "P3>1C": (-8, 4, "right")}
    for r in bl.sort_values("actual", ascending=False).head(6).itertuples(index=False):
        key = r.lane.split(":")[0]
        dx, dy, ha = offsets.get(key, (5, 5, "left"))
        ax.annotate(names.get(key, r.lane.replace(":", " ")), (max(r.actual, 12), max(r.plan, 12)), fontsize=7.5,
                    color=INK2, xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Recorded by EIA (kt per month)")
    ax.set_ylabel("Planned by our model (kt per month)")
    v = s["validation"]
    ax.set_title(f"Each dot is one route between regions · correlation {v['lane_corr_pearson']:.2f}", loc="left", fontsize=10, color=INK2,
                 fontweight="normal")
    save(fig, "validation_flows.png")


def fig_carbon(s):
    sw = pd.read_csv(os.path.join(RES, "carbon_sweep.csv"))
    a = sw.groupby("lambda_usd_per_t")[["cost_total_musd", "co2_total_kt", "co2_refining_kt",
                                        "co2_transport_kt", "co2_imports_kt"]].mean()
    base = a.iloc[0]
    dc = (a["cost_total_musd"] - base["cost_total_musd"])
    de = (a["co2_total_kt"] / base["co2_total_kt"] - 1) * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(de, dc, color=GREEN, marker="o", ms=8, mec=BG)
    for lam, x, y in zip(a.index, de, dc):
        if lam >= 100:
            ax.annotate(f"${lam:g}/t", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8.5, color=INK2)
    small = [l for l in a.index if l < 100]
    if small:
        ax.annotate("$" + ", ".join(f"{l:g}" for l in small) + "/t", (de.loc[small].mean(), dc.loc[small].mean()),
                    xytext=(0, 26), textcoords="offset points", fontsize=8.5, color=INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.set_xlabel("Change in CO₂ vs the cost-only plan (%)")
    ax.set_ylabel("Extra cost per quarter ($ M, excl. carbon charge)")
    save(fig, "carbon_tradeoff.png")


def fig_backtest(s):
    bo = pd.read_csv(os.path.join(RES, "backtest_by_origin.csv"), parse_dates=["origin"])
    base = bo[bo["strategy"] == "MILP P50"].set_index("origin")
    strat = ["BAU rule", "MILP P90", "MILP P50 + XGB risk", "MILP P50 + GAT risk", "MILP P50 + CO2 $100/t",
             "Perfect information"]
    cols = [ORANGE, YELLOW, BLUE, VIOLET, GREEN, INK3]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    for j, (col, lab, scale) in enumerate([("real_cost_total_musd", "Cost vs the standard optimiser ($ M per quarter)", 1),
                                           ("real_co2_total_kt", "CO₂ vs the standard optimiser (kt per quarter)", 1)]):
        ax = axes[j]
        rng = np.random.default_rng(0)
        for i, (st_, c) in enumerate(zip(strat, cols)):
            x = bo[bo["strategy"] == st_].set_index("origin")[col]
            dlt = (x - base[col]).dropna().to_numpy() * scale
            bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.plot([lo, hi], [i, i], color=c, lw=2.5, solid_capstyle="round")
            ax.scatter(dlt.mean(), i, color=c, s=55, edgecolor=BG, zorder=3)
            ax.text(hi, i, f"  {dlt.mean():+,.0f}", fontsize=9, color=INK2, va="center")
        ax.axvline(0, color=INK3, lw=1)
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.18 * (x1 - x0))
        nice = ["Rule of thumb", "Plan for high demand", "Risk-aware (XGBoost)", "Risk-aware (graph)",
                "With $100/t CO₂ price", "Perfect foresight"]
        ax.set_yticks(range(len(strat)), nice if j == 0 else [""] * len(strat))
        if j == 1:
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "backtest_deltas.png")


def fig_monte_carlo():
    p = os.path.join(RES, "monte_carlo_2019-08.csv")
    if not os.path.exists(p):
        return
    mc = pd.read_csv(p)
    order = ["BAU rule", "MILP P50", "MILP P90", "MILP P50 + XGB risk"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (st_, c) in enumerate(zip(order, [ORANGE, BLUE, YELLOW, AQUA])):
        v = mc[mc["strategy"] == st_]["cost_total_musd"] / 1000
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=14, color=c, alpha=0.8)
        ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color=INK, lw=2)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("Realised cost, Aug–Oct 2019 ($ bn)")
    ax.grid(axis="x", visible=False)
    save(fig, "monte_carlo.png")


def fig_utilization(t):
    w = t["utilization_weekly"]
    fig, ax = plt.subplots(figsize=(12, 3.0))
    for padd, c, lab in [("P3", BLUE, "Gulf Coast (PADD 3)"), ("P1", AQUA, "East Coast (PADD 1)"),
                         ("P2", ORANGE, "Midwest (PADD 2)")]:
        s_ = w[(w["padd"] == padd) & (w["week"] >= "2016-01-01")]
        ax.plot(s_["week"], s_["utilization_pct"], color=c, lw=1.2, label=lab)
    ax.set_ylabel("Refinery utilisation, %")
    ax.legend(fontsize=9, ncol=3, loc="lower left")
    save(fig, "weekly_utilization.png")


if __name__ == "__main__":
    s, t = load()
    fig_network_map(s, t)
    fig_forecast_examples(s, t)
    fig_forecast_by_product(s)
    fig_risk_timeline(s)
    fig_risk_metrics(s)
    fig_validation(s)
    fig_carbon(s)
    fig_backtest(s)
    fig_monte_carlo()
    fig_utilization(t)
