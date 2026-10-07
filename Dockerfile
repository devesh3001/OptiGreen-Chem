# OptiGreen-Chem dashboard (Stage 2, real EIA data).
# Lean runtime image: the dashboard loads the trained models from results/stage2 and solves
# the MILP live with HiGHS; PyTorch is not needed (it is only used to train the GAT).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first so code changes do not invalidate this layer
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Code, the extracted EIA tables, trained models and results (see .dockerignore)
COPY . .

# Run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT', '8501'))" || exit 1

# Render and most hosts inject $PORT; locally it defaults to 8501
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
