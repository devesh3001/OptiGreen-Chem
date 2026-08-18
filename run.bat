@echo off
echo Starting OptiGreen-Chem Dashboard...
set PYTHONPATH=src
python -m streamlit run app/streamlit_app.py
pause
