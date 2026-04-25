"""
preprocessing.py — Data cleaning, imputation, encoding, scaling, and SMOTE.
"""

import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODEL_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where missing fraction exceeds threshold.

    Args:
        df: Input DataFrame.
        threshold: Max allowed missing fraction.

    Returns:
        DataFrame with high-missing columns removed.
    """
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} high-missing columns: {drop_cols[:5]}...")
        df = df.drop(columns=drop_cols)
    return df


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaNs: median for numeric, mode for categorical.

    Args:
        df: DataFrame after column drops.

    Returns:
        Fully imputed DataFrame.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    logger.info(f"Imputed {len(num_cols)} numeric + {len(cat_cols)} categorical columns.")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals: OrdinalEncoder (<=5 unique) or get_dummies (>5).

    Args:
        df: DataFrame with categorical columns.

    Returns:
        Fully numeric DataFrame.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card  = [c for c in cat_cols if df[c].nunique() <= 5]
    high_card = [c for c in cat_cols if df[c].nunique() >  5]

    if low_card:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[low_card] = enc.fit_transform(df[low_card].astype(str))

    if high_card:
        df = pd.get_dummies(df, columns=high_card, drop_first=True)

    bool_cols = df.select_dtypes(include=bool).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    logger.info(f"Encoded {len(low_card)} ordinal + {len(high_card)} one-hot columns.")
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train; transform both splits.

    Args:
        X_train: Training features.
        X_test:  Test features.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    logger.info("StandardScaler applied.")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to oversample the minority class.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Resampled (X_train, y_train).
    """
    if not HAS_SMOTE:
        logger.warning("SMOTE skipped — imbalanced-learn not installed.")
        return X_train, y_train

    before = dict(zip(*np.unique(y_train, return_counts=True)))
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info(f"SMOTE: {before} → {after}")
    return X_res, y_res


def preprocess_pipeline(df: pd.DataFrame):
    """Full preprocessing: clean → impute → encode → split → scale → SMOTE.

    Args:
        df: Raw DataFrame (post feature engineering).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    logger.info("── Preprocessing Pipeline Start ──")

    df = drop_high_missing(df, threshold=0.5)
    df = impute_features(df)
    df = encode_categoricals(df)

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])

    non_num = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)

    feature_names = X.columns.tolist()
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train_s, X_test_s, scaler = scale_features(
        pd.DataFrame(X_train, columns=feature_names),
        pd.DataFrame(X_test,  columns=feature_names),
    )

    X_train_res, y_train_res = apply_smote(X_train_s, y_train)

    os.makedirs(MODEL_PATH, exist_ok=True)
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")

    logger.info("── Preprocessing Pipeline Complete ──")
    return X_train_res, X_test_s, y_train_res, y_test, scaler, feature_names
