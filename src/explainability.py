"""
explainability.py — SHAP-based model explainability and single-applicant explanations.
"""

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.warning("shap not installed — explainability features unavailable.")

from src.config import PLOTS_PATH, PLOT_DPI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _check_shap():
    if not HAS_SHAP:
        raise ImportError("Install shap: pip install shap")


# ── 1. compute_shap_values ────────────────────────────────────────────────────

def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray,
                         sample_size: int = 1000):
    """Compute SHAP values using TreeExplainer on a sample of X_test.

    Args:
        model:       Fitted tree-based model (LightGBM, XGBoost, RF).
        X_train:     Training data (used as background for explainer).
        X_test:      Test data (SHAP values computed on a sample).
        sample_size: Max rows to explain (for speed).

    Returns:
        (explainer, shap_values, X_sample).
    """
    _check_shap()
    n = min(sample_size, len(X_test))
    idx = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]

    logger.info(f"Computing SHAP values for {n} samples…")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info("SHAP computation complete.")
    return explainer, shap_values, X_sample


# ── 2. plot_shap_summary ──────────────────────────────────────────────────────

def plot_shap_summary(shap_values: np.ndarray, X_sample: np.ndarray,
                       feature_names: Optional[List[str]] = None) -> None:
    """SHAP beeswarm summary plot.

    Args:
        shap_values:   2-D SHAP values array.
        X_sample:      Feature matrix for the sample.
        feature_names: Column names.
    """
    _check_shap()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=PLOT_DPI)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    out = os.path.join(PLOTS_PATH, "shap_summary.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 3. plot_shap_bar ──────────────────────────────────────────────────────────

def plot_shap_bar(shap_values: np.ndarray, X_sample: np.ndarray,
                   feature_names: Optional[List[str]] = None) -> None:
    """SHAP global feature importance bar chart.

    Args:
        shap_values:   2-D SHAP values array.
        X_sample:      Feature matrix for the sample.
        feature_names: Column names.
    """
    _check_shap()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=PLOT_DPI)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, plot_size=None)
    plt.tight_layout()
    out = os.path.join(PLOTS_PATH, "shap_bar.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 4. plot_shap_waterfall ────────────────────────────────────────────────────

def plot_shap_waterfall(explainer, X_sample: np.ndarray, idx: int = 0,
                         feature_names: Optional[List[str]] = None) -> None:
    """SHAP waterfall plot for a single prediction.

    Args:
        explainer:     SHAP TreeExplainer.
        X_sample:      Sample feature matrix.
        idx:           Row index within X_sample to explain.
        feature_names: Column names.
    """
    _check_shap()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    try:
        sv = explainer(X_sample)
        # Handle multi-output
        if len(sv.shape) == 3:
            sv = sv[:, :, 1]
        shap.waterfall_plot(sv[idx], show=False)
        out = os.path.join(PLOTS_PATH, f"shap_waterfall_{idx}.png")
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {out}")
    except Exception as e:
        logger.warning(f"Waterfall plot failed: {e}")


# ── 5. plot_shap_force ────────────────────────────────────────────────────────

def plot_shap_force(explainer, shap_values: np.ndarray, X_sample: np.ndarray,
                     idx: int = 0, feature_names: Optional[List[str]] = None) -> None:
    """SHAP force plot saved as HTML.

    Args:
        explainer:     SHAP TreeExplainer.
        shap_values:   2-D SHAP values array.
        X_sample:      Sample feature matrix.
        idx:           Row index to explain.
        feature_names: Column names.
    """
    _check_shap()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    try:
        base = float(explainer.expected_value[1]) if isinstance(
            explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
        html = shap.force_plot(base, shap_values[idx],
                               X_sample[idx], feature_names=feature_names, matplotlib=False)
        out = os.path.join(PLOTS_PATH, f"shap_force_{idx}.html")
        shap.save_html(out, html)
        logger.info(f"Saved → {out}")
    except Exception as e:
        logger.warning(f"Force plot failed: {e}")


# ── 6. plot_shap_dependence ───────────────────────────────────────────────────

def plot_shap_dependence(shap_values: np.ndarray, X_sample: np.ndarray,
                          feature: str = "EXT_SOURCE_2",
                          interaction_feature: str = "DEBT_TO_INCOME",
                          feature_names: Optional[List[str]] = None) -> None:
    """SHAP dependence plot showing interaction effect.

    Args:
        shap_values:         2-D SHAP values.
        X_sample:            Feature matrix.
        feature:             Primary feature to plot.
        interaction_feature: Feature used for colour coding.
        feature_names:       Column names.
    """
    _check_shap()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    try:
        names = feature_names or list(range(X_sample.shape[1]))
        if feature not in names:
            logger.warning(f"Feature '{feature}' not found — skipping dependence plot.")
            return
        interact = interaction_feature if interaction_feature in names else "auto"
        fig, ax = plt.subplots(figsize=(10, 6), dpi=PLOT_DPI)
        shap.dependence_plot(feature, shap_values, X_sample,
                             feature_names=names, interaction_index=interact,
                             ax=ax, show=False)
        plt.tight_layout()
        out = os.path.join(PLOTS_PATH, f"shap_dependence_{feature}.png")
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {out}")
    except Exception as e:
        logger.warning(f"Dependence plot failed: {e}")


# ── 7. get_top_reasons ────────────────────────────────────────────────────────

def get_top_reasons(
    explainer,
    X_single_row: np.ndarray,
    feature_names: List[str],
    n: int = 5,
) -> List[dict]:
    """Explain a single applicant in plain English.

    Args:
        explainer:     SHAP TreeExplainer.
        X_single_row:  1-D or (1, n_features) feature array for one applicant.
        feature_names: Feature column names.
        n:             Number of top reasons to return.

    Returns:
        List of dicts: [{feature, value, shap_value, direction}, ...]
        direction is "increases risk" or "decreases risk".
    """
    _check_shap()
    row = X_single_row.reshape(1, -1)
    sv  = explainer.shap_values(row)
    if isinstance(sv, list):
        sv = sv[1]
    sv_1d = sv.flatten()

    top_idx = np.argsort(np.abs(sv_1d))[::-1][:n]
    reasons = []
    for i in top_idx:
        reasons.append({
            "feature":    feature_names[i],
            "value":      float(row[0, i]),
            "shap_value": float(sv_1d[i]),
            "direction":  "increases risk" if sv_1d[i] > 0 else "decreases risk",
        })
    return reasons
