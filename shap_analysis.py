"""SHAP explainability helpers for child malnutrition prediction models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _ensure_directory(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _extract_shap_array(shap_values) -> np.ndarray:
    values = None
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            values = np.array(shap_values[1])
        else:
            values = np.array(shap_values[0])
    elif hasattr(shap_values, "values"):
        values = np.array(shap_values.values)
    else:
        values = np.array(shap_values)

    if values.ndim == 3 and values.shape[1] in [1, 2]:
        values = values[:, -1, :]
    return values


def compute_shap_values(model, X):
    """Build a SHAP explainer and compute values for a feature matrix."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    return explainer, shap_values


def get_top_features(shap_values, feature_names: list[str], top_n: int = 10) -> list[str]:
    values = _extract_shap_array(shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    return importance.head(top_n).index.tolist()


def get_sample_top_features(shap_values, feature_names: list[str], sample_index: int, top_n: int = 5) -> list[str]:
    values = _extract_shap_array(shap_values)
    if sample_index < 0 or sample_index >= values.shape[0]:
        raise IndexError(f"Sample index {sample_index} is out of range for SHAP values with shape {values.shape}")
    sample_abs = np.abs(values[sample_index])
    top_idx = np.argsort(sample_abs)[-top_n:][::-1]
    return [feature_names[i] for i in top_idx]


def get_sample_top_features_batch(shap_values, feature_names: list[str], top_n: int = 5) -> list[list[str]]:
    values = _extract_shap_array(shap_values)
    abs_values = np.abs(values)
    top_idx = np.argpartition(abs_values, -top_n, axis=1)[:, -top_n:]
    result: list[list[str]] = []
    for row_indices, row_values in zip(top_idx, abs_values):
        sorted_order = np.argsort(row_values[row_indices])[::-1]
        chosen = [feature_names[idx] for idx in row_indices[sorted_order]]
        result.append(chosen)
    return result


def save_shap_summary_plot(shap_values, X, feature_names: list[str], output_path: str, max_display: int = 20, sample_size: int = 3000) -> None:
    _ensure_directory(output_path)
    X_arr = np.asarray(X)
    values_arr = _extract_shap_array(shap_values)

    if X_arr.shape[0] > sample_size:
        idx = np.random.default_rng(42).choice(X_arr.shape[0], size=sample_size, replace=False)
        X_plot = X_arr[idx]
        shap_values_plot = values_arr[idx]
    else:
        X_plot = X_arr
        shap_values_plot = values_arr

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_plot, feature_names=feature_names, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_importance_plot(shap_values, feature_names: list[str], output_path: str, top_n: int = 20) -> None:
    _ensure_directory(output_path)
    values = _extract_shap_array(shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    importance.tail(top_n).plot.barh(ax=ax, color="#2a7fb8")
    ax.set_title("SHAP Feature Importance (mean |SHAP|)")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_waterfall_plot(shap_values, feature_names: list[str], sample_index: int, output_path: str, max_display: int = 12) -> None:
    _ensure_directory(output_path)
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[sample_index], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
