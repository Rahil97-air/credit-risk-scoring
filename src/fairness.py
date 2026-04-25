"""
fairness.py — Group fairness audit for credit risk model.
"""

import os
import logging
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.config import PLOTS_PATH, REPORTS_PATH, PLOT_DPI, PLOT_FIGSIZE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. compute_group_metrics ──────────────────────────────────────────────────

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_feature: pd.Series,
) -> pd.DataFrame:
    """Compute fairness metrics per group in a sensitive feature.

    Args:
        y_true:            Ground-truth binary labels.
        y_pred:            Binary predictions.
        y_prob:            Predicted probabilities.
        sensitive_feature: Series of group labels (same index as y_true).

    Returns:
        DataFrame with one row per group and fairness metrics.
    """
    rows = []
    for group in sensitive_feature.unique():
        mask = sensitive_feature == group
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_prob[mask]

        if len(yt) == 0:
            continue

        # Approval rate: model predicted "no default" → approved
        approval_rate = float((yp == 0).mean())
        # Default rate among approved
        approved_mask = yp == 0
        def_rate_approved = float(yt[approved_mask].mean()) if approved_mask.sum() > 0 else 0.0
        # TPR (recall for default class)
        tpr = float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
        # FPR
        fpr = float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))
        # AUC
        try:
            group_auc = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else np.nan
        except Exception:
            group_auc = np.nan

        rows.append({
            "group":                    str(group),
            "n_samples":                int(mask.sum()),
            "approval_rate":            round(approval_rate, 4),
            "default_rate_approved":    round(def_rate_approved, 4),
            "TPR":                      round(tpr, 4),
            "FPR":                      round(fpr, 4),
            "AUC":                      round(group_auc, 4) if not np.isnan(group_auc) else None,
        })

    return pd.DataFrame(rows).sort_values("approval_rate", ascending=False)


# ── 2. plot_fairness_comparison ───────────────────────────────────────────────

def plot_fairness_comparison(
    group_metrics_df: pd.DataFrame,
    metric: str = "approval_rate",
    feature_name: str = "Feature",
) -> None:
    """Bar chart comparing a fairness metric across groups.

    Args:
        group_metrics_df: Output of compute_group_metrics().
        metric:           Column name to visualise.
        feature_name:     Label for the sensitive feature (used in title/filename).
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    df = group_metrics_df.copy()
    overall_avg = df[metric].mean()

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    colors = ["#e74c3c" if v < overall_avg * 0.9 else "#2ecc71" for v in df[metric]]
    ax.bar(df["group"].astype(str), df[metric] * 100, color=colors, edgecolor="white", width=0.55)
    ax.axhline(overall_avg * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Overall avg ({overall_avg*100:.1f}%)")

    ax.set_title(f"Fairness — {metric.replace('_', ' ').title()} by {feature_name}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=12)
    ax.set_xlabel(feature_name, fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()

    safe_name = feature_name.replace(" ", "_").lower()
    out = os.path.join(PLOTS_PATH, f"fairness_{safe_name}_{metric}.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 3. run_fairness_audit ─────────────────────────────────────────────────────

def run_fairness_audit(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Run fairness audit on gender, education type, and age bucket.

    Args:
        df:     Original DataFrame (pre-preprocessing) with raw column values.
        y_true: Ground-truth labels (aligned with df rows).
        y_pred: Binary predictions.
        y_prob: Predicted probabilities.

    Returns:
        Full audit DataFrame saved to outputs/reports/fairness_audit.csv.
    """
    sensitive_features = {
        "CODE_GENDER":         "Gender",
        "NAME_EDUCATION_TYPE": "Education Type",
    }

    # Age bucket (derived from AGE_YEARS or DAYS_BIRTH)
    if "AGE_YEARS" in df.columns:
        age = df["AGE_YEARS"]
    elif "DAYS_BIRTH" in df.columns:
        age = np.abs(df["DAYS_BIRTH"]) / 365.25
    else:
        age = None

    all_frames: List[pd.DataFrame] = []

    def _run(feature_series: pd.Series, name: str):
        feature_series = feature_series.reset_index(drop=True)
        gdf = compute_group_metrics(y_true, y_pred, y_prob, feature_series)
        gdf.insert(0, "sensitive_feature", name)
        plot_fairness_comparison(gdf, metric="approval_rate", feature_name=name)

        # Flag disparities
        max_rate = gdf["approval_rate"].max()
        min_rate = gdf["approval_rate"].min()
        if (max_rate - min_rate) > 0.10:
            logger.warning(
                f"⚠  Disparity detected in {name}: "
                f"max={max_rate:.2%}, min={min_rate:.2%}, gap={max_rate-min_rate:.2%}"
            )
            print(f"  ⚠  Fairness disparity in {name}: gap = {(max_rate-min_rate)*100:.1f}%")
        all_frames.append(gdf)

    print("\n── Fairness Audit ───────────────────────────")
    for col, label in sensitive_features.items():
        if col in df.columns:
            _run(df[col].astype(str), label)
        else:
            logger.info(f"Column '{col}' not found — skipping.")

    if age is not None:
        bins   = [0, 30, 45, 60, 200]
        labels = ["<30", "30-45", "45-60", "60+"]
        age_bucket = pd.cut(
            pd.Series(age).reset_index(drop=True),
            bins=bins, labels=labels, right=False
        ).astype(str)
        _run(age_bucket, "Age Group")

    os.makedirs(REPORTS_PATH, exist_ok=True)
    if all_frames:
        audit_df = pd.concat(all_frames, ignore_index=True)
        out = os.path.join(REPORTS_PATH, "fairness_audit.csv")
        audit_df.to_csv(out, index=False)
        logger.info(f"Fairness audit saved → {out}")
        return audit_df

    return pd.DataFrame()
