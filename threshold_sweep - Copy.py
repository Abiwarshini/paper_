from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import importlib.util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_stunting_enhanced.pkl"
SRC_PATH = PROJECT_ROOT / "src" / "02_run_models.py"

spec = importlib.util.spec_from_file_location("run_models", str(SRC_PATH))
run_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_models)

with open(MODEL_PATH, "rb") as f:
    metadata = pickle.load(f)
model = metadata["model"]
scaler = metadata["scaler"]

X, y, _ = run_models.load_data("stunting", "enhanced")
X_scaled = scaler.transform(X)
y_prob = model.predict_proba(X_scaled)[:, 1]

best = None
results = []
def compute_metrics(threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }

for t in np.arange(0.20, 0.81, 0.01):
    metrics = compute_metrics(float(t))
    results.append(metrics)
    if best is None or metrics["f1_score"] > best["f1_score"]:
        best = metrics

print("default_threshold_metrics", compute_metrics(0.5))
print("best_f1_threshold_metrics", best)
print("Done")
