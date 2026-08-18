# OptiGreen-Chem

**An AI-Driven Multi-Modal Supply Chain Intelligence and Sustainable Optimization Platform for Chemical Manufacturing**

OptiGreen-Chem is a rigorous AI/ML and optimization system designed to manage a simulated chemical manufacturing supply chain. It predicts demand uncertainty, assesses risks of stockouts, and runs mathematical optimizations to balance cost, carbon emissions, and service levels.

## Features

- **Probabilistic Demand Forecasting:** Uses Quantile XGBoost to forecast demand with confidence intervals (P10, P50, P90).
- **Risk Intelligence:** Employs Graph Neural Networks (GAT) to capture spatial supply chain dependencies and predict region-level stockout risks.
- **MILP Optimization Engine:** Uses Pyomo and the open-source HiGHS solver to perform multi-objective optimization (Cost vs. Carbon vs. Risk) under capacity constraints.
- **Interactive Dashboard:** A Streamlit web application providing a unified view of the entire intelligence pipeline.

## Project Structure

- `app/`: Streamlit dashboard (`streamlit_app.py`)
- `configs/`: YAML configuration files for the optimization engine, GNN, and dataset generators.
- `data/`: Contains synthetic raw data and processed prediction artifacts.
- `models/`: Saved model weights for XGBoost and GNN.
- `scripts/`: Python scripts to run individual pipeline phases (e.g., `run_phase2.py`, `run_phase3.py`).
- `src/optigreen/`: The core Python package containing modules for data generation, forecasting, risk modeling, and optimization.
- `tests/`: Pytest unit tests for the pipeline.

## Installation

1. Ensure you have Python 3.11+ installed.
2. Clone this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For Windows/Linux, it is highly recommended to install the CPU-only version of PyTorch to save memory, which the requirements file is configured to do).*

## Running the Dashboard

To launch the interactive Streamlit dashboard locally, you can use the provided batch script or run Streamlit directly:

```bash
# Run directly with Streamlit
python -m streamlit run app/streamlit_app.py
```

## Running the Pipeline

The project is structured into sequential phases. To re-run the intelligence pipeline from scratch and generate new predictions or optimization plans, execute the scripts in the `scripts/` directory:

1. **Data Synthesis:** (Phase 1 & 2) Generate the synthetic supply chain data.
2. **Forecasting:** (Phase 3) Train the Quantile XGBoost models and generate `pxgb_preds.csv`.
3. **Risk Intelligence:** (Phase 5) Train the GNN to compute spatial stockout risks.
4. **Optimization:** (Phase 4 & 6) Run the Pyomo/HiGHS optimizer to generate the multi-objective Pareto frontiers and Monte Carlo simulations. (These scripts output to `data/processed/` which the dashboard then visualizes).

## Environment Variables

When running tests or scripts manually, ensure the `src/` directory is in your Python path:
```bash
# Windows PowerShell
$env:PYTHONPATH="src"
python -m pytest
```
