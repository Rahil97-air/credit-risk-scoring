"""
scorecard.py — FICO-style credit score conversion (300–850 scale).
"""

import os
import logging
from math import log

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    SCORECARD_OFFSET, SCORECARD_FACTOR, PLOTS_PATH, REPORTS_PATH,
    PLOT_DPI, PLOT_FIGSIZE
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. probability_to_score ───────────────────────────────────────────────────

def probability_to_score(
    probability: float,
    offset: int = SCORECARD_OFFSET,
    factor: float = SCORECARD_FACTOR,
) -> int:
    """Convert default probability to a FICO-style credit score.

    Args:
        probability: Predicted default probability in [0, 1].
        offset:      Score at odds of 1:1 (default 600).
        factor:      Points-to-double-odds factor (default 20/ln(2)).

    Returns:
        Integer credit score clipped to [300, 850].
    """
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    odds  = (1 - probability) / probability
    score = offset + factor * log(odds)
    return int(np.clip(round(score), 300, 850))


def _vectorized_score(probabilities: np.ndarray) -> np.ndarray:
    """Vectorised version of probability_to_score for arrays.

    Args:
        probabilities: Array of predicted probabilities.

    Returns:
        Integer score array.
    """
    probs = np.clip(probabilities, 1e-6, 1 - 1e-6)
    odds  = (1 - probs) / probs
    scores = SCORECARD_OFFSET + SCORECARD_FACTOR * np.log(odds)
    return np.clip(np.round(scores), 300, 850).astype(int)


# ── 2. score_to_grade ─────────────────────────────────────────────────────────

def score_to_grade(score: int) -> str:
    """Map credit score to a letter grade with description.

    Args:
        score: Integer credit score (300–850).

    Returns:
        Grade string, e.g. "A — Excellent".
    """
    if score >= 750:
        return "A — Excellent"
    elif score >= 700:
        return "B — Good"
    elif score >= 650:
        return "C — Fair"
    elif score >= 600:
        return "D — Poor"
    else:
        return "E — High Risk"


# ── 3. add_scores_to_dataframe ────────────────────────────────────────────────

def add_scores_to_dataframe(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Add RISK_PROBABILITY, CREDIT_SCORE, CREDIT_GRADE columns to df.

    Args:
        df:            DataFrame to augment.
        probabilities: Predicted default probabilities (same length as df).

    Returns:
        Augmented DataFrame with three new columns.
    """
    df = df.copy()
    df["RISK_PROBABILITY"] = probabilities
    df["CREDIT_SCORE"]     = _vectorized_score(probabilities)
    df["CREDIT_GRADE"]     = df["CREDIT_SCORE"].apply(score_to_grade)
    return df


# ── 4. plot_score_distribution ────────────────────────────────────────────────

def plot_score_distribution(scores: np.ndarray, y_true: np.ndarray) -> None:
    """Histogram of credit scores split by actual default/no-default.

    Args:
        scores: Array of integer credit scores.
        y_true: True binary labels.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    ax.hist(scores[y_true == 0], bins=60, alpha=0.6, color="#2ecc71",
            label="No Default (0)", density=True)
    ax.hist(scores[y_true == 1], bins=60, alpha=0.6, color="#e74c3c",
            label="Default (1)", density=True)

    for threshold, label in [(600, "600 — Poor"), (700, "700 — Good")]:
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1.2)
        ax.text(threshold + 2, ax.get_ylim()[1] * 0.95, label,
                fontsize=8, color="gray", rotation=90, va="top")

    ax.set_xlabel("Credit Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Credit Score Distribution by Default Status", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "score_distribution.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 5. generate_scorecard_report ──────────────────────────────────────────────

def generate_scorecard_report(df_with_scores: pd.DataFrame) -> None:
    """Save scorecard CSV and print grade-level summary.

    Args:
        df_with_scores: DataFrame with CREDIT_SCORE, CREDIT_GRADE, TARGET columns.
    """
    os.makedirs(REPORTS_PATH, exist_ok=True)
    out = os.path.join(REPORTS_PATH, "scorecard_report.csv")
    df_with_scores.to_csv(out, index=False)
    logger.info(f"Scorecard report saved → {out}")

    print("\n── Scorecard Summary by Grade ───────────────")
    if "CREDIT_GRADE" in df_with_scores.columns:
        grp = df_with_scores.groupby("CREDIT_GRADE")
        summary = grp["CREDIT_SCORE"].mean().rename("Avg Score").round(1)
        if "TARGET" in df_with_scores.columns:
            default_rate = grp["TARGET"].mean().rename("Default Rate %").mul(100).round(2)
            count = grp.size().rename("Count")
            table = pd.concat([count, summary, default_rate], axis=1).sort_index()
        else:
            count = grp.size().rename("Count")
            table = pd.concat([count, summary], axis=1).sort_index()
        print(table.to_string())
    print()
