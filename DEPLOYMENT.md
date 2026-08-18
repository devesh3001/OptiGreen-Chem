# OptiGreen-Chem Deployment Guide

This guide details how to deploy the OptiGreen-Chem Streamlit dashboard using Docker and Render.

## Prerequisites
- Docker Engine installed locally
- A Render account (for cloud deployment)
- The existing trained model artifacts must be present in the `models/` directory:
  - `pxgb_model_P10.json`
  - `pxgb_model_P50.json`
  - `pxgb_model_P90.json`
  - `xgb_risk_model.json`
  - `gat_model.pt`

## Local Docker Deployment

### 1. Build the Docker Image
Run the following command from the repository root:
```bash
docker build -t optigreen-chem .
```

### 2. Run the Container Locally
```bash
docker run -p 8501:8501 optigreen-chem
```
The application will be accessible at `http://localhost:8501`.

## Render Deployment

This repository includes a `render.yaml` configuration file for zero-touch deployment on Render.

### Steps to Deploy
1. Push this repository to GitHub.
2. Log into Render (https://dashboard.render.com).
3. Go to **Blueprints** and click **New Blueprint Instance**.
4. Connect your GitHub repository.
5. Render will automatically detect the `render.yaml` file and provision the web service.

### Environment Variables
The `render.yaml` file configures the necessary environment variables:
- `PORT=8501`
- `PYTHONPATH=/app/src`

Render will automatically inject a host environment and map `PORT:8501` to the public web interface.

## Troubleshooting

### "Missing prediction artifact" or "Missing model artifact"
**Cause:** A required `.csv` prediction file or `.json`/`.pt` model file is missing from the container.
**Solution:** Ensure `.dockerignore` is NOT ignoring the `data/synthetic`, `data/processed`, or `models` directories. Rebuild the image.

### Pyomo/HiGHS Solver Fails
**Cause:** The MILP optimization requires a solver. The `highspy` pip package comes bundled with the solver binary for most Linux platforms.
**Solution:** Verify `highspy>=1.7.0` is installed via `requirements.txt`.

### Streamlit Caching Errors
**Cause:** During development, cached elements might conflict.
**Solution:** In the top-right corner of the Streamlit app, click **Clear Cache**, or restart the Docker container.
