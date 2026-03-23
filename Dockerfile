# ── Base image ────────────────────────────────────────────────────────────────
# Using the slim variant keeps the image lean for production deployments.
FROM python:3.10-slim

# ── Build-time argument ───────────────────────────────────────────────────────
# RUN_ID is the MLflow Run ID passed in at build time, e.g.:
#   docker build --build-arg RUN_ID=<mlflow-run-id> -t my-model .
ARG RUN_ID

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="assignment5"
LABEL mlflow.run_id="${RUN_ID}"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install MLflow client (used to download the model at runtime) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir mlflow

# ── Download the model from the MLflow Tracking Server ───────────────────────
# In a real deployment this would use 'mlflow artifacts download'.
# Here we simulate it with an echo statement because we do not have a
# public MLflow server available at build time.
RUN echo "Downloading model for MLflow Run ID: ${RUN_ID}"

# ── Copy inference code ───────────────────────────────────────────────────────
COPY . .

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "train.py"]
