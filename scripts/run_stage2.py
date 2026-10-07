"""
Run the complete OptiGreen-Chem Stage-2 pipeline on real EIA data.

Usage
-----
    # 1) download the public EIA bulk file (55 MB, no key needed)
    #    https://www.eia.gov/opendata/bulk/PET.zip
    # 2) run
    python scripts/run_stage2.py --bulk path/to/PET.zip
    python scripts/run_stage2.py              # reuse data/real/eia/*.csv
    python scripts/run_stage2.py --quick      # 6 origins, ~5 min smoke test

Outputs go to results/stage2/ (metrics JSON, CSV tables, pickled models) and
are read by app/streamlit_app.py and scripts/make_stage2_figures.py.
"""
import argparse
import json
import os
import pickle
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from optigreen.pipeline import stage2 as s2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk", default=None, help="path to EIA PET.zip (optional if tables exist)")
    ap.add_argument("--tables", default=os.path.join(ROOT, "data", "real", "eia"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "stage2"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--reuse-models", action="store_true",
                    help="reuse forecasters / risk models fitted by a previous run (cache in --out)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    d = s2.load_data(a.bulk, a.tables)
    summary = {"data": s2.data_summary(d)}
    s2.log(f"data: {summary['data']['regions']} regions, {summary['data']['districts']} districts")

    years = [2018, 2019] if a.quick else [2018, 2019, 2020, 2021]
    origins = (pd.to_datetime(["2018-03-01", "2018-09-01", "2019-02-01", "2019-08-01", "2019-11-01", "2019-12-01"])
               if a.quick else pd.date_range("2018-01-01", "2021-12-01", freq="MS"))
    origins = [o for o in origins if o + pd.DateOffset(months=2) <= pd.Timestamp("2022-03-01")]

    # ---- forecasting ----
    cache = os.path.join(a.out, "_fit_cache.pkl")
    cached = pickle.load(open(cache, "rb")) if (a.reuse_models and os.path.exists(cache)) else None
    fcs = cached["fcs"] if cached else s2.fit_forecasters(d["panel"], years)
    fev = s2.forecast_eval(d["panel"], fcs, list(pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")))
    summary["forecast"] = fev["metrics"]
    fev["predictions"].to_csv(os.path.join(a.out, "forecast_predictions.csv.gz"), index=False)
    summary["forecast_models"] = {y: {"best_trees": f.best_iter, "conformal_adjust": f.conformal_q,
                                      "train_end": str(f.train_end.date())} for y, f in fcs.items()}
    s2.log("forecast evaluation done")

    # ---- risk ----
    rows = s2.risk_rows(d["labels"])
    roff = s2.risk_offline(rows)
    summary["risk"] = {"metrics": roff["metrics"], "ap_ci": roff["ap_ci"], "auc_ci": roff["auc_ci"],
                       "split": roff["split"], "gat_best_epochs": roff["gat_epochs"],
                       "graph": s2.graph_checks(d["labels"])}
    roff["predictions"].to_csv(os.path.join(a.out, "risk_test_predictions.csv"), index=False)
    pd.DataFrame(roff["attention"], index=s2.om.DISTRICTS, columns=s2.om.DISTRICTS).to_csv(
        os.path.join(a.out, "gat_attention.csv"))
    risk_by_year = cached["risk"] if cached else {y: s2.fit_risk_for_year(rows, y) for y in years}
    if not cached:
        with open(cache, "wb") as fh:
            pickle.dump({"fcs": fcs, "risk": risk_by_year}, fh)
    s2.log("risk models done")
    d["labels"].to_csv(os.path.join(a.out, "outage_labels.csv"), index=False)

    # ---- backtest ----
    bt = s2.run_backtest(d, list(origins), fcs, risk_by_year, a.out)
    bt["by_origin"].to_csv(os.path.join(a.out, "backtest_by_origin.csv"), index=False)
    summ = s2.summarise_backtest(bt["by_origin"])
    summ.to_csv(os.path.join(a.out, "backtest_summary.csv"), index=False)
    cis = {}
    for other in ["BAU rule", "MILP P90", "MILP P50 + XGB risk", "MILP P50 + GAT risk", "MILP P50 + CO2 $100/t"]:
        for col in ["real_cost_total_musd", "real_cost_supply_chain_musd", "real_fill_rate", "real_co2_total_kt",
                    "real_unmet_kt"]:
            cis[f"{other} minus MILP P50 | {col}"] = s2.paired_ci(bt["by_origin"], other, "MILP P50", col)
    summary["backtest_ci"] = cis
    bt["flows"].to_csv(os.path.join(a.out, "plan_flows_milp_p50.csv.gz"), index=False)
    bt["crude"].to_csv(os.path.join(a.out, "plan_crude_milp_p50.csv"), index=False)
    s2.log("backtest done")

    # ---- validation ----
    val = s2.flow_validation(d, bt["flows"], bt["crude"])
    summary["validation"] = {k: v for k, v in val.items() if not isinstance(v, pd.DataFrame)}
    val["by_lane"].to_csv(os.path.join(a.out, "validation_by_lane.csv"))
    val["pairs"].to_csv(os.path.join(a.out, "validation_lane_month_pairs.csv"), index=False)
    val["util_pairs"].to_csv(os.path.join(a.out, "validation_utilization.csv"), index=False)

    # ---- carbon sweep ----
    sweep_or = [o for o in origins if o.year == 2019][:12 if not a.quick else 2]
    sw = s2.carbon_sweep(d, sweep_or, fcs[2019])
    sw.to_csv(os.path.join(a.out, "carbon_sweep.csv"), index=False)
    s2.log("carbon sweep done")

    # ---- Monte Carlo (Stage-3 preview) ----
    if not a.skip_mc:
        mc = s2.monte_carlo(d, pd.Timestamp("2019-08-01"), fcs[2019], risk_by_year[2019], n=8 if a.quick else 30)
        mc.to_csv(os.path.join(a.out, "monte_carlo_2019-08.csv"), index=False)

    # ---- headline plan (for the dashboard) ----
    head = s2.run_origin(d, pd.Timestamp("2019-08-01"), fcs[2019], risk_by_year[2019],
                         strategies={"MILP P50": s2.STRATEGIES["MILP P50"]})
    with open(os.path.join(a.out, "headline_plan_2019-08.pkl"), "wb") as fh:
        pickle.dump(head["plans"]["MILP P50"], fh)
    import gzip
    with gzip.open(os.path.join(a.out, "models.pkl.gz"), "wb", compresslevel=9) as fh:
        pickle.dump({"forecasters": fcs, "risk_by_year": {y: {"xgboost": m["xgboost"]} for y, m in risk_by_year.items()}}, fh)

    summary["runtime_min"] = (time.time() - t_start) / 60
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(s2.to_jsonable(summary), fh, indent=2, default=str)
    s2.log(f"all done in {summary['runtime_min']:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
