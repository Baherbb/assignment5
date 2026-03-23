"""
check_threshold.py
------------------
Reads the MLflow Run ID from model_info.txt, fetches the
'accuracy' metric from the MLflow Tracking Server, and exits
with a non-zero status code if the accuracy is below the
required threshold (0.85).

This script is called by the 'deploy' job in the GitHub Actions
workflow.  When it exits with code 1 the workflow marks the job
as FAILED, which prevents the mock Docker build from running.
"""

import sys
import os
import mlflow

# ── Configuration ───────────────────────────────────────────────────────────
MODEL_INFO_FILE = "model_info.txt"
ACCURACY_THRESHOLD = 0.99
METRIC_NAME = "accuracy"

# ── 1. Read Run ID from the artifact passed by the validate job ─────────────
if not os.path.exists(MODEL_INFO_FILE):
    print(f"ERROR: '{MODEL_INFO_FILE}' not found. Did the validate job run?")
    sys.exit(1)

with open(MODEL_INFO_FILE, "r") as f:
    run_id = f.read().strip()

if not run_id:
    print("ERROR: model_info.txt is empty. Cannot retrieve Run ID.")
    sys.exit(1)

print(f"Retrieved Run ID: {run_id}")

# ── 2. Connect to MLflow and fetch the metric ───────────────────────────────
# MLFLOW_TRACKING_URI is set as an environment variable from the GitHub
# Actions secret; mlflow.set_tracking_uri() reads it automatically.
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("ERROR: MLFLOW_TRACKING_URI environment variable is not set.")
    sys.exit(1)

mlflow.set_tracking_uri(tracking_uri)

client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"ERROR: Could not retrieve run '{run_id}' from MLflow: {e}")
    sys.exit(1)

# ── 3. Extract accuracy metric ───────────────────────────────────────────────
metrics = run.data.metrics
if METRIC_NAME not in metrics:
    print(
        f"ERROR: Metric '{METRIC_NAME}' not found in run '{run_id}'. "
        f"Available metrics: {list(metrics.keys())}"
    )
    sys.exit(1)

accuracy = metrics[METRIC_NAME]
print(f"Accuracy from MLflow: {accuracy:.4f}")
print(f"Required threshold  : {ACCURACY_THRESHOLD}")

# ── 4. Apply threshold gate ──────────────────────────────────────────────────
if accuracy < ACCURACY_THRESHOLD:
    print(
        f"\n❌ PIPELINE HALTED: Accuracy {accuracy:.4f} is below the "
        f"required threshold of {ACCURACY_THRESHOLD}. "
        f"Deployment blocked."
    )
    sys.exit(1)   # Non-zero exit → GitHub Actions marks the step as FAILED

print(
    f"\n✅ Threshold check PASSED: Accuracy {accuracy:.4f} ≥ "
    f"{ACCURACY_THRESHOLD}. Proceeding to deployment."
)
sys.exit(0)
