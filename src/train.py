"""
train.py — Model training pipeline: Logistic, Random Forest, LightGBM + Optuna.
"""

import os
import logging
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.config import (
    RANDOM_STATE, CV_FOLDS, MODEL_PATH, LGB_PARAMS, OPTUNA_TRIALS
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore")


# ── 1. Baseline Logistic Regression ──────────────────────────────────────────

def train_baseline_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[LogisticRegression, float]:
    """Fit a balanced Logistic Regression and report AUC.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test:  Test features.
        y_test:  Test labels.

    Returns:
        (fitted_model, auc_score).
    """
    logger.info("Training Baseline Logistic Regression…")
    model = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"\n── Logistic Regression ──────────────────────")
    print(f"  AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, (probs >= 0.5).astype(int),
                                 target_names=["No Default", "Default"]))
    return model, auc


# ── 2. Random Forest ─────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[RandomForestClassifier, np.ndarray]:
    """Fit Random Forest with Stratified K-Fold CV.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test:  Test features.
        y_test:  Test labels.

    Returns:
        (fitted_model, cv_auc_scores).
    """
    logger.info("Training Random Forest…")
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, max_depth=12
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                scoring="roc_auc", n_jobs=-1)

    model.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n── Random Forest ────────────────────────────")
    print(f"  CV AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    return model, cv_scores


# ── 3. Optuna LightGBM Tuning ─────────────────────────────────────────────────

def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Hyperparameter search for LightGBM using Optuna.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Best hyperparameter dict found by Optuna.
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed — returning default LGB_PARAMS.")
        return LGB_PARAMS.copy()

    logger.info(f"Running Optuna ({OPTUNA_TRIALS} trials)…")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 5.0),
            "n_estimators":      500,
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
            "verbosity":         -1,
            "n_jobs":            -1,
        }
        aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xtr, Xval = X_train[train_idx], X_train[val_idx]
            ytr, yval = y_train[train_idx], y_train[val_idx]
            m = lgb.LGBMClassifier(**params)
            m.fit(Xtr, ytr,
                  eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
            aucs.append(roc_auc_score(yval, m.predict_proba(Xval)[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({"class_weight": "balanced", "random_state": RANDOM_STATE,
                 "verbosity": -1, "n_jobs": -1})

    print(f"\n── Optuna Best Params ───────────────────────")
    for k, v in best.items():
        print(f"  {k:<25}: {v}")
    print(f"  Best CV AUC: {study.best_value:.4f}")
    return best


# ── 4. Final LightGBM ────────────────────────────────────────────────────────

def train_final_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[dict] = None,
    feature_names: Optional[list] = None,
) -> Tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray]:
    """Train final LightGBM with OOF predictions and early stopping.

    Args:
        X_train:       Training features.
        y_train:       Training labels.
        X_test:        Test features.
        y_test:        Test labels.
        params:        LightGBM hyperparameters (uses config defaults if None).
        feature_names: Column names for the model.

    Returns:
        (fitted_model, oof_predictions, test_predictions).
    """
    if params is None:
        params = LGB_PARAMS.copy()

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    best_model = None
    best_auc   = -1.0

    logger.info("Training Final LightGBM with OOF…")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        val_prob = model.predict_proba(Xval)[:, 1]
        oof_preds[val_idx] = val_prob
        fold_auc = roc_auc_score(yval, val_prob)
        test_preds += model.predict_proba(X_test)[:, 1] / CV_FOLDS

        if fold_auc > best_auc:
            best_auc   = fold_auc
            best_model = model

        logger.info(f"  Fold {fold}/{CV_FOLDS} — AUC: {fold_auc:.4f}")

    oof_auc  = roc_auc_score(y_train, oof_preds)
    test_auc = roc_auc_score(y_test, test_preds)

    print(f"\n── Final LightGBM ───────────────────────────")
    print(f"  OOF AUC : {oof_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")

    os.makedirs(MODEL_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_PATH, "lgbm_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"LightGBM model saved → {model_path}")

    return best_model, oof_preds, test_preds


# ── 5. Full Training Pipeline ────────────────────────────────────────────────

def run_full_training_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
    skip_optuna: bool = False,
) -> Dict[str, tuple]:
    """Run all models and return comparison dict.

    Args:
        X_train:       Training features.
        y_train:       Training labels.
        X_test:        Test features.
        y_test:        Test labels.
        feature_names: Feature column names.
        skip_optuna:   If True, skip Optuna and use default LGB params.

    Returns:
        Dict mapping model_name → (model, auc_score).
    """
    results: Dict[str, tuple] = {}

    # Logistic Regression
    lr_model, lr_auc = train_baseline_logistic(X_train, y_train, X_test, y_test)
    results["Logistic Regression"] = (lr_model, lr_auc)

    # Random Forest
    rf_model, rf_cv = train_random_forest(X_train, y_train, X_test, y_test)
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    results["Random Forest"] = (rf_model, rf_auc)

    # Optuna tuning
    if skip_optuna:
        best_params = LGB_PARAMS.copy()
        logger.info("Optuna skipped — using default LGB_PARAMS.")
    else:
        best_params = tune_lightgbm_optuna(X_train, y_train)

    # Final LightGBM
    lgbm_model, oof, test_preds = train_final_lightgbm(
        X_train, y_train, X_test, y_test,
        params=best_params, feature_names=feature_names
    )
    lgbm_auc = roc_auc_score(y_test, test_preds)
    results["LightGBM"] = (lgbm_model, lgbm_auc)

    # Comparison table
    print(f"\n{'='*50}")
    print(f"  Model Comparison")
    print(f"{'='*50}")
    print(f"  {'Model':<25} {'AUC-ROC':>10}")
    print(f"  {'-'*35}")
    for name, (_, auc) in results.items():
        print(f"  {name:<25} {auc:>10.4f}")
    print(f"{'='*50}\n")

    return results
