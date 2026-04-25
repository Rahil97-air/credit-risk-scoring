"""
threshold_optimizer.py — Business-oriented threshold analysis for credit decisions.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from src.config import PLOTS_PATH, REPORTS_PATH, PLOT_DPI, LGD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. compute_threshold_metrics ──────────────────────────────────────────────

def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute classification metrics at each threshold.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted default probabilities.
        thresholds: Array of thresholds; defaults to np.arange(0.1, 0.9, 0.02).

    Returns:
        DataFrame with columns: threshold, precision, recall, f1, f2,
        approval_rate, defaults_caught_pct.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.02)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_approved = int((y_pred == 0).sum())   # approved = predicted no-default
        n_total    = len(y_true)
        n_defaults = int(y_true.sum())

        # Defaults caught = predicted positive (rejected) who actually defaulted
        defaults_caught = int(((y_pred == 1) & (y_true == 1)).sum())

        rows.append({
            "threshold":          round(t, 4),
            "precision":          precision_score(y_true, y_pred, zero_division=0),
            "recall":             recall_score(y_true, y_pred, zero_division=0),
            "f1":                 f1_score(y_true, y_pred, zero_division=0),
            "f2":                 fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "approval_rate":      n_approved / n_total,
            "defaults_caught_pct": defaults_caught / max(n_defaults, 1),
        })

    return pd.DataFrame(rows)


# ── 2. plot_threshold_analysis ────────────────────────────────────────────────

def plot_threshold_analysis(metrics_df: pd.DataFrame) -> None:
    """2×2 subplot dashboard of threshold tradeoffs.

    Args:
        metrics_df: Output of compute_threshold_metrics().
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    opt_t = get_optimal_threshold(metrics_df, metric="f2")
    t     = metrics_df["threshold"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=PLOT_DPI)

    def _vline(ax):
        ax.axvline(opt_t, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Optimal ({opt_t:.2f})")

    # (a) Precision vs Recall
    ax = axes[0, 0]
    ax.plot(t, metrics_df["precision"], label="Precision", color="#3498db", linewidth=2)
    ax.plot(t, metrics_df["recall"],    label="Recall",    color="#e74c3c", linewidth=2)
    _vline(ax)
    ax.set_title("Precision & Recall vs Threshold"); ax.legend(); ax.grid(alpha=0.3)

    # (b) F1 and F2
    ax = axes[0, 1]
    ax.plot(t, metrics_df["f1"], label="F1", color="#2ecc71", linewidth=2)
    ax.plot(t, metrics_df["f2"], label="F2 (β=2)", color="#9b59b6", linewidth=2)
    _vline(ax)
    ax.set_title("F1 & F2 vs Threshold"); ax.legend(); ax.grid(alpha=0.3)

    # (c) Approval Rate
    ax = axes[1, 0]
    ax.plot(t, metrics_df["approval_rate"] * 100, color="#f39c12", linewidth=2)
    _vline(ax)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_title("Approval Rate vs Threshold"); ax.legend(); ax.grid(alpha=0.3)

    # (d) Defaults Caught
    ax = axes[1, 1]
    ax.plot(t, metrics_df["defaults_caught_pct"] * 100, color="#e74c3c", linewidth=2)
    _vline(ax)
    ax.set_ylabel("Defaults Caught (%)")
    ax.set_title("Defaults Caught % vs Threshold"); ax.legend(); ax.grid(alpha=0.3)

    for ax in axes.flatten():
        ax.set_xlabel("Threshold")

    plt.suptitle("Threshold Optimization Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "threshold_analysis.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 3. compute_business_tradeoff_table ────────────────────────────────────────

def compute_business_tradeoff_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray,
    lgd: float = LGD,
) -> pd.DataFrame:
    """Expected loss and revenue tradeoff at fixed thresholds.

    Args:
        y_true:       Ground-truth labels.
        y_prob:       Predicted default probabilities.
        loan_amounts: Loan EAD (Exposure at Default) for each record.
        lgd:          Loss Given Default fraction.

    Returns:
        DataFrame with business metrics per threshold.
    """
    thresholds = [0.2, 0.3, 0.4, 0.45, 0.5, 0.6]
    rows = []

    for t in thresholds:
        approved_mask  = y_prob < t          # predicted non-default → approved
        n_approved     = int(approved_mask.sum())
        defaults_in    = int((approved_mask & (y_true == 1)).sum())
        pd_approved    = y_prob[approved_mask]
        ead_approved   = loan_amounts[approved_mask]
        el             = float((pd_approved * lgd * ead_approved).sum())
        reject_rate    = 1 - n_approved / max(len(y_true), 1)

        rows.append({
            "threshold":             t,
            "approved_loans":        n_approved,
            "defaults_in_approved":  defaults_in,
            "expected_loss":         round(el, 2),
            "revenue_lost_pct":      round(reject_rate * 100, 2),
        })

    df = pd.DataFrame(rows)
    os.makedirs(REPORTS_PATH, exist_ok=True)
    out = os.path.join(REPORTS_PATH, "business_tradeoff.csv")
    df.to_csv(out, index=False)

    print("\n── Business Tradeoff Table ──────────────────")
    print(df.to_string(index=False))
    logger.info(f"Saved → {out}")
    return df


# ── 4. get_optimal_threshold ──────────────────────────────────────────────────

def get_optimal_threshold(metrics_df: pd.DataFrame, metric: str = "f2") -> float:
    """Return the threshold that maximises the given metric.

    Args:
        metrics_df: Output of compute_threshold_metrics().
        metric:     Column name to optimise (e.g. 'f1', 'f2').

    Returns:
        Optimal threshold as float.
    """
    idx = metrics_df[metric].idxmax()
    opt = float(metrics_df.loc[idx, "threshold"])
    logger.info(f"Optimal threshold ({metric}): {opt:.4f} → {metric}={metrics_df.loc[idx, metric]:.4f}")
    return opt
