"""
expected_loss.py — PD × LGD × EAD expected loss calculation and portfolio reporting.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import LGD, PLOTS_PATH, REPORTS_PATH, PLOT_DPI, PLOT_FIGSIZE
from src.scorecard import score_to_grade, _vectorized_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 1. compute_expected_loss ──────────────────────────────────────────────────

def compute_expected_loss(
    probabilities: np.ndarray,
    loan_amounts: np.ndarray,
    lgd: float = LGD,
) -> np.ndarray:
    """Compute individual Expected Loss: EL = PD × LGD × EAD.

    Args:
        probabilities: Predicted default probability per loan.
        loan_amounts:  Exposure at Default (EAD) per loan.
        lgd:           Loss Given Default fraction (default 0.45).

    Returns:
        Array of expected loss values.
    """
    return probabilities * lgd * loan_amounts


# ── 2. compute_portfolio_metrics ──────────────────────────────────────────────

def compute_portfolio_metrics(
    probabilities: np.ndarray,
    loan_amounts: np.ndarray,
    y_true: np.ndarray = None,
    lgd: float = LGD,
) -> dict:
    """Aggregate portfolio-level risk metrics.

    Args:
        probabilities: Predicted PD per loan.
        loan_amounts:  Loan EAD per loan.
        y_true:        Actual default labels (optional).
        lgd:           Loss Given Default.

    Returns:
        Dict of portfolio metrics.
    """
    el = compute_expected_loss(probabilities, loan_amounts, lgd)
    metrics = {
        "total_expected_loss":   float(el.sum()),
        "avg_pd":                float(probabilities.mean()),
        "weighted_avg_pd":       float(np.average(probabilities, weights=loan_amounts)),
        "portfolio_size":        int(len(probabilities)),
        "total_exposure":        float(loan_amounts.sum()),
    }
    if y_true is not None:
        actual_defaults = int(y_true.sum())
        actual_loss     = float((y_true * loan_amounts * lgd).sum())
        metrics["actual_defaults"]    = actual_defaults
        metrics["actual_loss"]        = actual_loss
        metrics["actual_loss_rate"]   = round(actual_loss / max(loan_amounts.sum(), 1), 6)

    return metrics


# ── 3. plot_expected_loss_distribution ───────────────────────────────────────

def plot_expected_loss_distribution(
    el_series: np.ndarray,
    loan_amounts: np.ndarray,
    probabilities: np.ndarray = None,
) -> None:
    """Histogram of individual EL + scatter of loan amount vs EL.

    Args:
        el_series:     Per-loan expected loss values.
        loan_amounts:  Per-loan EAD.
        probabilities: PD values (for colour-coded scatter).
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=PLOT_DPI)

    # (a) EL histogram
    ax = axes[0]
    ax.hist(el_series, bins=60, color="#e74c3c", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Expected Loss (₹)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Individual Expected Losses", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # (b) Scatter: loan amount vs EL
    ax = axes[1]
    if probabilities is not None:
        sc = ax.scatter(loan_amounts, el_series, c=probabilities,
                        cmap="RdYlGn_r", alpha=0.3, s=5)
        plt.colorbar(sc, ax=ax, label="Predicted PD")
    else:
        ax.scatter(loan_amounts, el_series, color="#3498db", alpha=0.3, s=5)
    ax.set_xlabel("Loan Amount (₹)", fontsize=12)
    ax.set_ylabel("Expected Loss (₹)", fontsize=12)
    ax.set_title("Loan Amount vs Expected Loss", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_PATH, "expected_loss_dist.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ── 4. generate_portfolio_risk_report ────────────────────────────────────────

def generate_portfolio_risk_report(
    probabilities: np.ndarray,
    loan_amounts: np.ndarray,
    y_true: np.ndarray = None,
    lgd: float = LGD,
) -> pd.DataFrame:
    """Full portfolio risk report: metrics + grade-level breakdown.

    Args:
        probabilities: Predicted PD per loan.
        loan_amounts:  Loan EAD.
        y_true:        Actual default labels (optional).
        lgd:           Loss Given Default.

    Returns:
        Grade-level summary DataFrame.
    """
    el = compute_expected_loss(probabilities, loan_amounts, lgd)
    metrics = compute_portfolio_metrics(probabilities, loan_amounts, y_true, lgd)

    print("\n── Portfolio Risk Metrics ───────────────────")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v:,.4f}" if isinstance(v, float) else f"  {k:<25}: {v:,}")

    # Grade-level breakdown
    scores = _vectorized_score(probabilities)
    grades = [score_to_grade(s) for s in scores]
    df = pd.DataFrame({
        "grade":        grades,
        "pd":           probabilities,
        "el":           el,
        "loan_amount":  loan_amounts,
    })
    if y_true is not None:
        df["actual_default"] = y_true

    grp = df.groupby("grade")
    summary = pd.DataFrame({
        "count":    grp.size(),
        "avg_PD":   grp["pd"].mean().round(4),
        "total_EL": grp["el"].sum().round(2),
        "avg_loan": grp["loan_amount"].mean().round(2),
    })
    if y_true is not None:
        summary["actual_default_rate"] = grp["actual_default"].mean().round(4)

    summary = summary.sort_index()

    os.makedirs(REPORTS_PATH, exist_ok=True)
    out = os.path.join(REPORTS_PATH, "portfolio_risk_report.csv")
    summary.to_csv(out)
    logger.info(f"Portfolio risk report saved → {out}")

    plot_expected_loss_distribution(el, loan_amounts, probabilities)

    print("\n── Portfolio Summary by Grade ───────────────")
    print(summary.to_string())
    return summary
