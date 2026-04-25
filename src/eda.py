"""
eda.py — Exploratory Data Analysis functions for the Credit Risk Scoring System.

All plots are saved to PLOTS_PATH; nothing is shown interactively.
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import logging
from datetime import datetime
from typing import Optional, List

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False
    logging.warning("missingno not installed — missing-value matrix will be skipped.")

# ── Internal ──────────────────────────────────────────────────────────────────
from src.config import PLOTS_PATH, TARGET_COL, PLOT_DPI, PLOT_FIGSIZE

# ── Module-level logger ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Palette ───────────────────────────────────────────────────────────────────
_PALETTE = {0: "#2ecc71", 1: "#e74c3c"}   # green = no default, red = default


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset and print summary statistics.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        Loaded DataFrame.
    """
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    print(f"\n{'='*60}")
    print(f"  Dataset loaded  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"\n  Dtype summary:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {str(dtype):<12} : {count} columns")
    print(f"{'='*60}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. plot_target_distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Bar chart of TARGET class distribution with percentage labels.

    Args:
        df: Raw DataFrame containing TARGET column.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    counts = df[TARGET_COL].value_counts().sort_index()
    pcts   = counts / counts.sum() * 100
    labels = ["No Default (0)", "Default (1)"]
    colors = [_PALETTE[0], _PALETTE[1]]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="white", linewidth=1.5)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=12, fontweight="bold"
        )

    ax.set_title("Target Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.18)
    sns.despine(ax=ax)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "target_distribution.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. plot_missing_values
# ─────────────────────────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame) -> None:
    """Visualise missingness — missingno matrix + top-30 bar chart.

    Args:
        df: Raw DataFrame.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    top30 = missing_pct[missing_pct > 0].head(30)

    # ── Missingno matrix (sample for speed) ──────────────────────────────────
    if HAS_MISSINGNO and not top30.empty:
        sample = df[top30.index].sample(min(5000, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(14, 6), dpi=PLOT_DPI)
        msno.matrix(sample, ax=ax, sparkline=False, color=(0.18, 0.55, 0.90))
        ax.set_title("Missing Value Matrix (sample of 5 000 rows)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(PLOTS_PATH, "missing_matrix.png")
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {out}")

    # ── Horizontal bar chart ──────────────────────────────────────────────────
    if top30.empty:
        logger.info("No missing values found — skipping missing bar chart.")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(top30) * 0.35)), dpi=PLOT_DPI)
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#3498db"
              for v in top30.values]
    ax.barh(top30.index[::-1], top30.values[::-1] * 100, color=colors[::-1], edgecolor="white")
    ax.axvline(50, color="red", linestyle="--", linewidth=1.2, label="50% threshold")
    ax.set_xlabel("Missing %", fontsize=12)
    ax.set_title("Top-30 Columns by Missing %", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "missing_values.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. plot_numeric_distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> None:
    """Histograms with KDE colored by TARGET, for the top numeric columns.

    Args:
        df:   DataFrame with TARGET column.
        cols: Explicit list of columns; if None, auto-selects top 12 by variance.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]

    if cols is None:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        cols = variances.head(12).index.tolist()

    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), dpi=PLOT_DPI)
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        for target_val, color in _PALETTE.items():
            subset = df.loc[df[TARGET_COL] == target_val, col].dropna()
            label = "No Default" if target_val == 0 else "Default"
            ax.hist(subset, bins=40, alpha=0.5, color=color, density=True, label=label)
            try:
                subset.plot.kde(ax=ax, color=color, linewidth=1.5)
            except Exception:
                pass
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Target", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "numeric_distributions.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. plot_correlation_matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap of the top-20 features most correlated with TARGET.

    Args:
        df: DataFrame with TARGET column.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    numeric_df = df.select_dtypes(include=np.number)
    corr_with_target = numeric_df.corr()[TARGET_COL].abs().sort_values(ascending=False)
    top_features = corr_with_target.head(21).index.tolist()  # includes TARGET itself

    corr_matrix = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12), dpi=PLOT_DPI)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn_r", center=0, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax, square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Matrix — Top 20 Features vs TARGET", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "correlation_matrix.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_class_imbalance_analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_imbalance_analysis(df: pd.DataFrame) -> None:
    """Default rate broken down by income bins, education type, and contract type.

    Args:
        df: Raw DataFrame.
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=PLOT_DPI)

    # ── (a) Income bins ───────────────────────────────────────────────────────
    ax = axes[0]
    if "AMT_INCOME_TOTAL" in df.columns:
        income_bins = pd.qcut(df["AMT_INCOME_TOTAL"], q=5, duplicates="drop")
        rate = df.groupby(income_bins, observed=True)[TARGET_COL].mean() * 100
        rate.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
        ax.set_title("Default Rate by Income Quintile", fontsize=12, fontweight="bold")
        ax.set_xlabel("Income Quintile")
        ax.set_ylabel("Default Rate (%)")
        ax.tick_params(axis="x", rotation=30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "AMT_INCOME_TOTAL not found", ha="center", va="center")

    # ── (b) Education type ────────────────────────────────────────────────────
    ax = axes[1]
    if "NAME_EDUCATION_TYPE" in df.columns:
        rate = df.groupby("NAME_EDUCATION_TYPE")[TARGET_COL].mean() * 100
        rate.sort_values().plot(kind="barh", ax=ax, color="#9b59b6", edgecolor="white")
        ax.set_title("Default Rate by Education Type", fontsize=12, fontweight="bold")
        ax.set_xlabel("Default Rate (%)")
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.1f}%",
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha="left", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NAME_EDUCATION_TYPE not found", ha="center", va="center")

    # ── (c) Contract type ─────────────────────────────────────────────────────
    ax = axes[2]
    if "NAME_CONTRACT_TYPE" in df.columns:
        rate = df.groupby("NAME_CONTRACT_TYPE")[TARGET_COL].mean() * 100
        rate.plot(kind="bar", ax=ax, color="#e67e22", edgecolor="white")
        ax.set_title("Default Rate by Contract Type", fontsize=12, fontweight="bold")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Default Rate (%)")
        ax.tick_params(axis="x", rotation=15)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "NAME_CONTRACT_TYPE not found", ha="center", va="center")

    plt.suptitle("Class Imbalance Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(PLOTS_PATH, "class_imbalance_analysis.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. run_full_eda
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame) -> dict:
    """Run all EDA steps in sequence and return a summary dict.

    Args:
        df: Raw DataFrame.

    Returns:
        Summary dict with n_rows, n_cols, missing_pct, default_rate.
    """
    print(f"\n{'='*60}")
    print("  Starting Full EDA Pipeline")
    print(f"{'='*60}\n")

    logger.info("Step 1/6 — Target Distribution")
    plot_target_distribution(df)

    logger.info("Step 2/6 — Missing Values")
    plot_missing_values(df)

    logger.info("Step 3/6 — Numeric Distributions")
    plot_numeric_distributions(df)

    logger.info("Step 4/6 — Correlation Matrix")
    plot_correlation_matrix(df)

    logger.info("Step 5/6 — Class Imbalance Analysis")
    plot_class_imbalance_analysis(df)

    summary = {
        "n_rows":        int(df.shape[0]),
        "n_cols":        int(df.shape[1]),
        "missing_pct":   round(df.isnull().mean().mean() * 100, 2),
        "default_rate":  round(df[TARGET_COL].mean() * 100, 2),
    }

    logger.info("Step 6/6 — EDA Complete")
    print(f"\n{'='*60}")
    print("  EDA Summary")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:<20} : {v}")
    print(f"{'='*60}\n")

    return summary
