# Deploying the OptiGreen-Chem dashboard

The dashboard (`app/streamlit_app.py`) runs on the committed Stage-2 outputs:

| What it reads | Where | Size |
|---|---|---|
| Extracted EIA tables | `data/real/eia/*.csv.gz` | 2.6 MB |
| Trained forecasters and risk models | `results/stage2/models.pkl.gz` | 12 MB |
| Back-test, validation, sweep and plan results | `results/stage2/*.csv`, `summary.json`, `headline_plan_2019-08.pkl` | ~4 MB |
| State boundaries for the map | `assets/us_states_naturalearth_50m.geojson` | 0.3 MB |

Nothing is trained at start-up. The Optimization tab solves the MILP live with HiGHS
(about 0.25 s per plan). The app needs **no PyTorch**: its runtime dependencies are in
`app/requirements.txt`. Peak memory is about **0.5 GB** with every tab opened.

Tested: Python 3.13, every tab plus a live solve, from a copy of exactly what the Docker
build context contains, and the Docker start command answering `/_stcore/health`.

---

## Option A · Streamlit Community Cloud (free, simplest)

1. Push the repository to GitHub (branch `stage2-real-data`, or `main` after merging).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `devesh3001/OptiGreen-Chem`
   - **Branch:** `stage2-real-data` (or `main`)
   - **Main file path:** `app/streamlit_app.py`
   - **Advanced settings → Python version:** 3.13 (3.12 also works)
4. Click **Deploy**. Community Cloud installs `app/requirements.txt` (it looks next to the
   entry point first) and reads `.streamlit/config.toml` from the repository root.

The first start takes a few minutes while packages install; later wake-ups are faster.

## Option B · Render (Docker, uses `render.yaml`)

1. Push the repository to GitHub.
2. In Render: **Blueprints** → **New Blueprint Instance** → connect the repository.
3. Render reads `render.yaml`: a Docker web service in Singapore on the 1 CPU / 2 GB plan,
   health check `/_stcore/health`, redeploy on every commit.
4. The 512 MB plans (`free`, `0.5c-512mb`) are cheaper but the app peaks near 512 MB and
   may restart; change `plan:` in `render.yaml` if you accept that.

## Option C · Run it locally

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py            # http://localhost:8501
```

or with Docker:

```bash
docker build -t optigreen-chem .
docker run -p 8501:8501 optigreen-chem
```

To regenerate the results the app reads (needs the full `requirements.txt`, including PyTorch):

```bash
pip install -r requirements.txt
python scripts/run_stage2.py
python scripts/make_stage2_figures.py
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "No results found" or empty tabs | `results/stage2/` is missing from the deployment. It is committed in the repository; make sure it was not excluded, or run `python scripts/run_stage2.py`. |
| Error while loading `models.pkl.gz` | The pickled models need the same major versions they were saved with: numpy 2, pandas 3, scikit-learn 1.9, xgboost 3 (pinned in `app/requirements.txt`). |
| Optimization tab: solver not found | `highspy` must be installed (it bundles the HiGHS binary for Linux, macOS and Windows). |
| App restarts on Render | Memory limit reached: use the 1 CPU / 2 GB plan. |
| "The graph attention network needs PyTorch" | Only raised when training the GAT (`scripts/run_stage2.py`); install PyTorch for that. |
