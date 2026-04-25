"""
config.py — All constants, paths, and hyperparameters for the Credit Risk Scoring System.

All modules import from here. Never hardcode values elsewhere.
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import os
from math import log

# ── File Paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = os.path.join("data", "raw", "application_train.csv")
MODEL_PATH      = os.path.join("models")
OUTPUT_PATH     = os.path.join("outputs")
PLOTS_PATH      = os.path.join("outputs", "plots")
REPORTS_PATH    = os.path.join("outputs", "reports")

# ── Experiment Settings ───────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ── Column Settings ───────────────────────────────────────────────────────────
TARGET_COL = "TARGET"

# ── LightGBM Default Hyperparameters ─────────────────────────────────────────
LGB_PARAMS: dict = {
    "num_leaves":        64,
    "learning_rate":     0.05,
    "n_estimators":      1000,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "class_weight":      "balanced",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbosity":         -1,
}

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 50

# ── Basel / Credit Risk Parameters ───────────────────────────────────────────
LGD = 0.45  # Loss Given Default (Basel II standard)

# ── FICO-Style Scorecard Parameters ──────────────────────────────────────────
SCORECARD_OFFSET = 600
SCORECARD_FACTOR = 20 / log(2)   # ≈ 28.85

# ── PSI Drift Thresholds ──────────────────────────────────────────────────────
PSI_THRESHOLD_WARNING  = 0.10
PSI_THRESHOLD_CRITICAL = 0.25

# ── Plot Defaults ─────────────────────────────────────────────────────────────
PLOT_DPI     = 150
PLOT_FIGSIZE = (12, 6)

# ── Ensure output directories exist at import time ───────────────────────────
for _dir in (MODEL_PATH, PLOTS_PATH, REPORTS_PATH):
    os.makedirs(_dir, exist_ok=True)
