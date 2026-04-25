"""
evaluate.py — Model evaluation: ROC, PR, confusion matrix, calibration, reports.
"""

import os
import logging
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)

from src.config import PLOTS_PATH, REPORTS_PATH, PLOT_DPI, PLOT_FIGSIZE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. plot_roc_curves ────────────────────────────────────────────────────────

def plot_roc_curves(
    models_dict: Dict[str, tuple],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Plot ROC curves for multiple models on the same axes.

    Args:
        models_dict: {name: (model, auc)} from training pipeline.
        X_test:      Test features.
        y_test:      Test labels.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random (AUC=0.50)")

    for (name, (model, _)), color in zip(models_dict.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.0,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "roc_curves.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 2. plot_precision_recall_curve ────────────────────────────────────────────

def plot_precision_recall_curve(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot Precision-Recall curve with average precision score.

    Args:
        model:  Fitted classifier.
        X_test: Test features.
        y_test: Test labels.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    probs = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax.step(rec, prec, where="post", color="#e74c3c", linewidth=2.0,
            label=f"LightGBM (AP = {ap:.4f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline (AP = {baseline:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "pr_curve.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 3. plot_confusion_matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Normalized confusion matrix heatmap.

    Args:
        model:     Fitted classifier.
        X_test:    Test features.
        y_test:    Test labels.
        threshold: Decision threshold for positive class.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    probs  = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    cm     = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=PLOT_DPI)
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"],
                linewidths=0.5, ax=ax, cbar_kws={"format": "%.0%%"})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "confusion_matrix.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 4. print_classification_report ────────────────────────────────────────────

def print_classification_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Print sklearn classification report, highlighting minority class.

    Args:
        model:     Fitted classifier.
        X_test:    Test features.
        y_test:    Test labels.
        threshold: Decision threshold.
    """
    probs  = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    print("\n── Classification Report ────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Default (0)", "Default (1)"]
    ))
    default_idx = 1
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["No Default", "Default"],
        output_dict=True
    )
    d = report_dict.get("Default", {})
    print(f"  ⚠  Minority class (Default):")
    print(f"     Precision : {d.get('precision', 0):.4f}")
    print(f"     Recall    : {d.get('recall', 0):.4f}")
    print(f"     F1-Score  : {d.get('f1-score', 0):.4f}")


# ── 5. plot_calibration_curve ─────────────────────────────────────────────────

def plot_calibration_curve(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot calibration curve for the given model.

    Args:
        model:  Fitted classifier.
        X_test: Test features.
        y_test: Test labels.
    """
    from sklearn.calibration import calibration_curve
    os.makedirs(PLOTS_PATH, exist_ok=True)
    probs = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")
    ax.plot(mean_pred, frac_pos, "s-", color="#e74c3c", linewidth=2, markersize=7,
            label="Model")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "calibration_curve.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 6. generate_full_evaluation_report ────────────────────────────────────────

def generate_full_evaluation_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    models_dict: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run all evaluation functions and save summary CSV.

    Args:
        model:         Best fitted classifier.
        X_test:        Test features.
        y_test:        Test labels.
        feature_names: Feature column names.
        models_dict:   All models for ROC comparison.

    Returns:
        Summary metrics DataFrame.
    """
    logger.info("Generating full evaluation report…")

    if models_dict:
        plot_roc_curves(models_dict, X_test, y_test)

    plot_precision_recall_curve(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)
    print_classification_report(model, X_test, y_test)
    plot_calibration_curve(model, X_test, y_test)

    probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    ap_score  = average_precision_score(y_test, probs)

    summary = pd.DataFrame([{
        "metric":  "AUC-ROC",
        "value":   round(auc_score, 4),
    }, {
        "metric":  "Average Precision",
        "value":   round(ap_score, 4),
    }, {
        "metric":  "Default Rate",
        "value":   round(float(y_test.mean()), 4),
    }])

    os.makedirs(REPORTS_PATH, exist_ok=True)
    out = os.path.join(REPORTS_PATH, "evaluation_report.csv")
    summary.to_csv(out, index=False)
    logger.info(f"Evaluation report saved → {out}")
    return summary
