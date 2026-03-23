"""
train.py
--------
Trains a simple classifier on the dataset pulled by DVC,
logs the run to MLflow, and writes the MLflow Run ID to
model_info.txt so the deploy job can retrieve it.
"""

import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# ── 1. Load dataset (replace with your dvc-tracked data path) ──────────────
# If a dvc-tracked CSV exists at data/dataset.csv we load it;
# otherwise we fall back to the built-in Iris dataset for demonstration.
DATA_PATH = "data/dataset.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
else:
    data = load_iris(as_frame=True)
    X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Train model ──────────────────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ── 3. Evaluate ─────────────────────────────────────────────────────────────
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# ── 4. Log to MLflow ────────────────────────────────────────────────────────
# MLFLOW_TRACKING_URI is injected as an environment variable from the
# GitHub Actions secret; mlflow.set_tracking_uri reads it automatically.
mlflow.set_experiment("assignment5-pipeline")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

# ── 5. Export Run ID ─────────────────────────────────────────────────────────
# Write ONLY the Run ID (no newline) so the deploy job can read it cleanly.
with open("model_info.txt", "w") as f:
    f.write(run_id)

print("model_info.txt written successfully.")
