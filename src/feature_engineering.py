"""
feature_engineering.py — Domain-driven feature creation for credit risk.

Call engineer_all_features(df) as the single entry point.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Flag columns created by each function (tracked for reporting)
_CREATED: list = []


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Element-wise division with zero-denominator protection.

    Args:
        a:    Numerator Series.
        b:    Denominator Series.
        fill: Value to use when denominator is zero.

    Returns:
        Quotient Series.
    """
    return np.where(b == 0, fill, a / b)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Financial Ratios
# ─────────────────────────────────────────────────────────────────────────────

def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Create key financial ratio features.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with added ratio columns.
    """
    new_cols = []

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["DEBT_TO_INCOME"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
        new_cols.append("DEBT_TO_INCOME")

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_TO_INCOME"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
        new_cols.append("ANNUITY_TO_INCOME")

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["CREDIT_TO_GOODS"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        new_cols.append("CREDIT_TO_GOODS")

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
        new_cols.append("INCOME_PER_PERSON")

    _CREATED.extend(new_cols)
    logger.info(f"Financial ratios added: {new_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Age / Employment Features
# ─────────────────────────────────────────────────────────────────────────────

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive age and employment duration features.

    DAYS_EMPLOYED = 365243 is a sentinel for 'unemployed/not applicable'.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with age/employment columns added.
    """
    new_cols = []

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25
        new_cols.append("AGE_YEARS")

    if "DAYS_EMPLOYED" in df.columns:
        days_emp = df["DAYS_EMPLOYED"].copy()
        days_emp = days_emp.where(days_emp != 365243, 0)   # sentinel → 0
        days_emp = np.minimum(days_emp, 0)                  # cap at 0 (negative means employed)
        df["EMPLOYMENT_YEARS"] = np.abs(days_emp) / 365.25
        new_cols.append("EMPLOYMENT_YEARS")

    if {"EMPLOYMENT_YEARS", "AGE_YEARS"}.issubset(df.columns):
        df["EMPLOYMENT_TO_AGE_RATIO"] = _safe_div(
            df["EMPLOYMENT_YEARS"], df["AGE_YEARS"]
        )
        new_cols.append("EMPLOYMENT_TO_AGE_RATIO")

    _CREATED.extend(new_cols)
    logger.info(f"Age/employment features added: {new_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Document Features
# ─────────────────────────────────────────────────────────────────────────────

def create_document_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sum all FLAG_DOCUMENT_* columns into a single DOCUMENT_COUNT feature.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with DOCUMENT_COUNT column added.
    """
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
        _CREATED.append("DOCUMENT_COUNT")
        logger.info(f"DOCUMENT_COUNT created from {len(doc_cols)} FLAG_DOCUMENT_* columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. External Score Features
# ─────────────────────────────────────────────────────────────────────────────

def create_external_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EXT_SOURCE_1/2/3 into summary statistics.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with EXT_SOURCE summary features added.
    """
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    new_cols = []

    if len(ext_cols) >= 1:
        ext_df = df[ext_cols]
        df["EXT_SOURCE_MEAN"] = ext_df.mean(axis=1)
        df["EXT_SOURCE_MIN"]  = ext_df.min(axis=1)
        df["EXT_SOURCE_STD"]  = ext_df.std(axis=1).fillna(0)
        new_cols.extend(["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_STD"])

        # Weighted combination (weights sum to 1.0)
        weights = {"EXT_SOURCE_1": 0.3, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.2}
        weighted = pd.Series(np.zeros(len(df)), index=df.index)
        total_w = 0.0
        for col, w in weights.items():
            if col in df.columns:
                weighted += df[col].fillna(0) * w
                total_w += w
        df["EXT_SOURCE_WEIGHTED"] = weighted / max(total_w, 1e-9)
        new_cols.append("EXT_SOURCE_WEIGHTED")

    _CREATED.extend(new_cols)
    logger.info(f"External score features added: {new_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Polynomial Interactions
# ─────────────────────────────────────────────────────────────────────────────

def create_polynomial_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiplicative interaction features.

    Args:
        df: DataFrame (should have financial ratios and external scores already).

    Returns:
        DataFrame with interaction columns added.
    """
    new_cols = []

    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["ANNUITY_x_CREDIT"] = df["AMT_ANNUITY"] * df["AMT_CREDIT"]
        new_cols.append("ANNUITY_x_CREDIT")

    if {"EXT_SOURCE_MEAN", "DEBT_TO_INCOME"}.issubset(df.columns):
        df["EXT_MEAN_x_DEBT_INCOME"] = df["EXT_SOURCE_MEAN"] * df["DEBT_TO_INCOME"]
        new_cols.append("EXT_MEAN_x_DEBT_INCOME")

    _CREATED.extend(new_cols)
    logger.info(f"Polynomial interactions added: {new_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. engineer_all_features (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame enriched with all engineered features.
    """
    _CREATED.clear()
    n_before = df.shape[1]

    df = create_financial_ratios(df)
    df = create_age_features(df)
    df = create_document_features(df)
    df = create_external_score_features(df)
    df = create_polynomial_interactions(df)

    n_after = df.shape[1]
    print(f"Added {n_after - n_before} new features. Total features: {n_after}")
    logger.info(f"Feature engineering complete. {n_before} → {n_after} columns.")
    return df
