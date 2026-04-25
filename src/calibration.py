"""
calibration.py — Probability calibration using isotonic regression.
"""

import os
import logging
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib

from src.config import MODEL_PATH, PLOTS_PATH, PLOT_DPI, PLOT_FIGSIZE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. calibrate_model ────────────────────────────────────────────────────────

def calibrate_model(base_model, X_train: np.ndarray, y_train: np.ndarray):
    """Wrap model with isotonic calibration and fit on training data.

    Args:
        base_model: Fitted sklearn-compatible classifier.
        X_train:    Training features.
        y_train:    Training labels.

    Returns:
        Fitted CalibratedClassifierCV model.
    """
    logger.info("Calibrating model with isotonic regression (cv=5)…")
    cal_model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=5)
    cal_model.fit(X_train, y_train)

    os.makedirs(MODEL_PATH, exist_ok=True)
    path = os.path.join(MODEL_PATH, "calibrated_model.pkl")
    joblib.dump(cal_model, path)
    logger.info(f"Calibrated model saved → {path}")
    return cal_model


# ── 2. compare_calibration ────────────────────────────────────────────────────

def compare_calibration(
    base_model,
    calibrated_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Plot calibration curves for base vs calibrated model.

    Args:
        base_model:       Original fitted model.
        calibrated_model: Calibrated wrapper model.
        X_test:           Test features.
        y_test:           Test labels.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

    models = {
        "Base Model":       base_model,
        "Calibrated Model": calibrated_model,
    }
    colors = ["#e74c3c", "#2ecc71"]
    errors = {}

    for (name, model), color in zip(models.items(), colors):
        probs = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", color=color, label=name, linewidth=2, markersize=6)
        errors[name] = float(np.mean(np.abs(frac_pos - mean_pred)))

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "calibration_comparison.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")

    print("\n── Calibration Errors ──────────────────────")
    for name, err in errors.items():
        print(f"  {name:<25}: {err:.4f}")


# ── 3. get_calibrated_probabilities ──────────────────────────────────────────

def get_calibrated_probabilities(calibrated_model, X: np.ndarray) -> np.ndarray:
    """Return calibrated positive-class probabilities.

    Args:
        calibrated_model: Fitted calibrated classifier.
        X:                Feature matrix.

    Returns:
        1-D array of predicted default probabilities.
    """
    return calibrated_model.predict_proba(X)[:, 1]
