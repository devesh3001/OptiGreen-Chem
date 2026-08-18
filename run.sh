#!/bin/bash
echo "Starting OptiGreen-Chem Dashboard..."
export PYTHONPATH=src
python -m streamlit run app/streamlit_app.py
