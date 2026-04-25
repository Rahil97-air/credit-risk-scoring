"""
drift_detection.py — Population Stability Index (PSI) drift monitoring.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PSI_THRESHOLD_WARNING, PSI_THRESHOLD_CRITICAL,
    PLOTS_PATH, REPORTS_PATH, PLOT_DPI
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. compute_psi_single ─────────────────────────────────────────────────────

def compute_psi_single(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index for one feature.

    PSI = Σ (actual% − expected%) × ln(actual% / expected%)

    Args:
        expected: Reference distribution (e.g. training data).
        actual:   Current distribution (e.g. production data).
        bins:     Number of quantile bins.

    Returns:
        PSI value as float.
    """
    epsilon = 1e-4
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / max(len(expected), 1)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / max(len(actual),   1)

    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


# ── 2. compute_psi_all_features ───────────────────────────────────────────────

def compute_psi_all_features(
    X_train: np.ndarray,
    X_production: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """Compute PSI for every feature column.

    Args:
        X_train:       Training (reference) feature matrix.
        X_production:  Production feature matrix.
        feature_names: Column names; auto-generated if None.

    Returns:
        DataFrame with columns: feature, psi, status.
    """
    n_cols = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_cols)]

    rows = []
    for i, name in enumerate(feature_names):
        try:
            col_exp = X_train[:, i]   if hasattr(X_train, "shape") else np.array(X_train)[:, i]
            col_act = X_production[:, i] if hasattr(X_production, "shape") else np.array(X_production)[:, i]
            psi = compute_psi_single(col_exp.astype(float), col_act.astype(float))
        except Exception:
            psi = 0.0

        if psi < PSI_THRESHOLD_WARNING:
            status = "Stable"
        elif psi < PSI_THRESHOLD_CRITICAL:
            status = "Warning"
        else:
            status = "Critical"

        rows.append({"feature": name, "psi": round(psi, 4), "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# ── 3. plot_psi_report ────────────────────────────────────────────────────────

def plot_psi_report(psi_df: pd.DataFrame) -> None:
    """Horizontal bar chart of PSI values coloured by stability status.

    Args:
        psi_df: Output of compute_psi_all_features().
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    top = psi_df.head(40)   # show at most top 40 for readability

    color_map = {"Stable": "#2ecc71", "Warning": "#f39c12", "Critical": "#e74c3c"}
    colors = [color_map.get(s, "gray") for s in top["status"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.35)), dpi=PLOT_DPI)
    ax.barh(top["feature"][::-1], top["psi"][::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(PSI_THRESHOLD_WARNING,  color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Warning ({PSI_THRESHOLD_WARNING})")
    ax.axvline(PSI_THRESHOLD_CRITICAL, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Critical ({PSI_THRESHOLD_CRITICAL})")
    ax.set_xlabel("PSI Value", fontsize=12)
    ax.set_title("Population Stability Index (PSI) Report", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "psi_report.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 4. simulate_drift ─────────────────────────────────────────────────────────

def simulate_drift(
    X_test: np.ndarray,
    drift_cols: list = None,
    drift_strength: float = 0.3,
    feature_names: list = None,
) -> np.ndarray:
    """Simulate production data drift by adding Gaussian noise.

    Args:
        X_test:         Original test feature matrix.
        drift_cols:     Indices or names of columns to drift; defaults to first 5.
        drift_strength: Noise multiplier (0.3 = 30% noise).
        feature_names:  Column names (used only for logging).

    Returns:
        Drifted feature matrix (copy of X_test with noise added).
    """
    X_drifted = X_test.copy().astype(float)
    n_cols = X_drifted.shape[1]

    if drift_cols is None:
        drift_indices = list(range(min(5, n_cols)))
    elif feature_names is not None and isinstance(drift_cols[0], str):
        drift_indices = [feature_names.index(c) for c in drift_cols if c in feature_names]
    else:
        drift_indices = drift_cols

    rng = np.random.default_rng(42)
    for i in drift_indices:
        noise = 1 + drift_strength * rng.standard_normal(len(X_drifted))
        X_drifted[:, i] *= noise

    names = [feature_names[i] for i in drift_indices] if feature_names else drift_indices
    logger.info(f"Simulated drift on columns: {names}")
    return X_drifted


# ── 5. run_drift_monitoring ───────────────────────────────────────────────────

def run_drift_monitoring(
    X_train: np.ndarray,
    X_production: np.ndarray,
    feature_names: list = None,
) -> pd.DataFrame:
    """Full drift monitoring pipeline: compute PSI, plot, save report.

    Args:
        X_train:       Reference (training) data.
        X_production:  Production data to monitor.
        feature_names: Column names.

    Returns:
        PSI DataFrame.
    """
    logger.info("Running drift monitoring…")
    psi_df = compute_psi_all_features(X_train, X_production, feature_names)
    plot_psi_report(psi_df)

    counts = psi_df["status"].value_counts()
    print("\n── Drift Monitoring Summary ─────────────────")
    for status in ["Stable", "Warning", "Critical"]:
        print(f"  {status:<10}: {counts.get(status, 0)} features")

    os.makedirs(REPORTS_PATH, exist_ok=True)
    out = os.path.join(REPORTS_PATH, "drift_report.csv")
    psi_df.to_csv(out, index=False)
    logger.info(f"Drift report saved → {out}")
    return psi_df
