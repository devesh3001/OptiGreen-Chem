# OptiGreen-Chem   https://optigreen-chem.streamlit.app/

**Supply-chain optimisation for refined chemical products using machine-learning demand
prediction, disruption-risk scoring and mixed-integer optimisation — on real public data.**

CHO301 · Section B · Stage 2 (Progress Review 2)

| Module | Method | Real data it uses |
|---|---|---|
| 1. Demand forecasting | Global quantile XGBoost, direct 1–3-month horizons, conformal P10–P90 | EIA prime-supplier sales, 50 states + DC × 5 products, 1993–2022 |
| 2. Risk scoring | Outage classifiers (logistic, XGBoost) and a graph attention network over 12 refining districts | EIA refinery utilisation & capacity 1985–2026 (outages: Katrina, Ike, Harvey, Uri, Ida, refinery fires) |
| 3. Optimisation | Pyomo + HiGHS MILP: crude runs, pipeline flows, **integer** tanker/barge/import cargoes, inventories, deliveries | EIA capacities, yields, stocks, inter-PADD flows by mode, imports/exports, prices |
| 4. Objective & evaluation | Cost + carbon price; plans re-scored against actual demand and actual outages; Monte Carlo | EIA refinery fuel use × EPA factors; Cefic/ECTA transport factors |

## Stage-2 results (all out of sample, from `results/stage2/summary.json`)

| Check | Result |
|---|---|
| Demand forecast, 2018–19, h = 1–3 months | WAPE **4.40 %** vs 5.75 % seasonal naive + growth (error −23.7 %, 95 % CI 20.7–27.0 %); P10–P90 hit rate 76 % (83 % by volume) |
| Demand forecast, 2020–21 (COVID) | WAPE 9.41 % vs 13.77 %; hit rate falls to 68 % |
| Outage risk, test 2017–2026 | Avg. precision: XGBoost 0.264, GAT 0.260, GAT without edges 0.266, seasonal climatology 0.210 (base rate 0.10) |
| Graph check | Linked districts co-fail 2.5× chance (unlinked 1.5×), but message passing adds no measurable skill |
| MILP size and speed | ≈4,650 variables (39–45 integer cargo variables), ≈1,740 constraints; 0.26 s mean HiGHS solve; 288/288 plans optimal |
| Plan vs EIA recorded flows (first plan month, 2018–21) | Lane volumes Pearson 0.98, Spearman 0.77; national crude run within 0.1 % of actual before 2020 |
| Back-test, 48 plans re-scored on actual demand and outages | Rule-based plan +$1.50 bn per quarter (CI 1.26–1.77) vs MILP P50; risk-aware MILP −$44 M (CI −68 to −22); P90 plan +$136 M and +1.63 Mt CO₂ |
| Carbon price $100/t | −1.27 Mt CO₂ per quarter (−2.0 %) for +$174 M (≈ $137/t); refinery fuel is 91 % of CO₂ in the boundary |

Fill rate is 100 % for every strategy at state-month level: national stocks and imports absorb
monthly shocks, so service-level differences need finer (weekly, terminal-level) data — Stage 3.

Why this dataset (and not DataCo or synthetic data): see [docs/DATASET_SELECTION.md](docs/DATASET_SELECTION.md).
What was wrong with the Stage-1 code and how it was fixed: [docs/STAGE2_AUDIT.md](docs/STAGE2_AUDIT.md).

## Quick start

```bash
pip install -r requirements.txt          # CPU torch is enough
export PYTHONPATH=src                     # Windows PowerShell: $env:PYTHONPATH="src"

# optional: refresh the data from the official EIA bulk file (55 MB, public domain)
#   https://www.eia.gov/opendata/bulk/PET.zip
python -m optigreen.data.eia PET.zip --out data/real/eia

python scripts/run_stage2.py --quick      # ~5 min smoke test
python scripts/run_stage2.py              # full run, ~40 min on 2 CPU cores
python scripts/make_stage2_figures.py     # figures for the review
python -m streamlit run app/streamlit_app.py
python -m pytest -q                       # 56 tests (pytest.ini sets the src path)
```

**Deploying the dashboard:** it needs only `app/requirements.txt` (no PyTorch) and the
committed results. Step-by-step for Streamlit Community Cloud (free), Render (Docker) and
local Docker: [DEPLOYMENT.md](DEPLOYMENT.md).

The extracted EIA tables (`data/real/eia/*.csv.gz`, 2.6 MB) are committed, so the pipeline
runs without downloading anything. `series_used.csv.gz` lists every EIA series id read.

## Network (real system)

* **12 plants** – EIA refining districts (East Coast, Appalachian, Indiana-Illinois-Kentucky,
  Minnesota-Wisconsin-Dakotas, Oklahoma-Kansas-Missouri, Texas Inland, Texas Gulf Coast,
  Louisiana Gulf Coast, North Louisiana-Arkansas, New Mexico, Rocky Mountain, West Coast).
* **10 hubs** – bulk-terminal systems on EIA stock areas (Boston, New York Harbor, Atlanta,
  Tampa, Chicago, Tulsa, Houston, Denver, Los Angeles, Seattle).
* **58 regions** – 50 states + DC + 7 pools for state volumes EIA withholds.
* **5 products** – motor gasoline, distillate, jet fuel, residual fuel oil, propane.
* **Lanes** – inter-PADD pipeline, tanker and barge lanes that carried product in the
  36 months before each plan, with capacities from the observed maxima.

## Repository layout

```
src/optigreen/data/eia.py            EIA bulk-file reader -> tidy tables
src/optigreen/data/realcase.py       demand / district / stock panels, outage labels, CO2 intensity
src/optigreen/data/network.py        nodes, coordinates, lanes, cost & emission factors (with sources)
src/optigreen/forecasting/demand_model.py   quantile XGBoost + conformal calibration + baselines
src/optigreen/risk/outage_model.py   outage features, logistic / XGBoost / GAT (pure PyTorch), metrics
src/optigreen/optimization/network_milp.py  planning MILP, recourse evaluation, BAU rule
src/optigreen/pipeline/stage2.py     backtest, carbon sweep, validation, Monte Carlo
scripts/run_stage2.py                end-to-end run -> results/stage2/
scripts/make_stage2_figures.py       review figures -> results/stage2/figures/
app/streamlit_app.py                 dashboard (live MILP solve)
tests/                               unit tests (incl. leakage and mass-balance tests)
```

The Stage-1 synthetic pipeline (`scripts/run_phase*.py`, `data/synthetic`, `models/`) is kept
as a test fixture; its known problems are fixed or documented in the audit.

## Data licence and citation

U.S. Energy Information Administration, *Petroleum and Other Liquids* bulk data (PET.zip),
public domain. Emission factors: U.S. EPA GHG Emission Factors Hub (2024); Cefic/ECTA,
*Guidelines for Measuring and Managing CO₂ Emission from Freight Transport Operations* (2011).
Pipeline tariff anchor: Colonial Pipeline FERC No. 99.95.0 (2026). State boundaries: Natural Earth (public domain).
