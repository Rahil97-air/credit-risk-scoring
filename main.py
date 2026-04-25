"""
main.py - Full pipeline orchestrator for the Credit Risk Scoring System.

Usage:
    python main.py                            # full pipeline
    python main.py --skip-eda                 # skip EDA plots
    python main.py --skip-optuna              # use default LightGBM params
    python main.py --sample 5000             # use only 5000 rows (for testing)
    python main.py --skip-eda --skip-optuna --sample 5000   # fast dev run
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys
import time
import warnings

# ── Force UTF-8 stdout on Windows to avoid cp1252 UnicodeEncodeError ─────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Internal ──────────────────────────────────────────────────────────────────
from src.config import (
    RAW_DATA_PATH, MODEL_PATH, PLOTS_PATH, REPORTS_PATH, RANDOM_STATE
)
from src.eda              import load_data, run_full_eda
from src.feature_engineering import engineer_all_features
from src.preprocessing    import preprocess_pipeline
from src.train            import run_full_training_pipeline
from src.calibration      import calibrate_model, compare_calibration, get_calibrated_probabilities
from src.evaluate         import generate_full_evaluation_report
from src.explainability   import (compute_shap_values, plot_shap_summary, plot_shap_bar,
                                   plot_shap_waterfall, plot_shap_force, plot_shap_dependence)
from src.scorecard        import add_scores_to_dataframe, plot_score_distribution, generate_scorecard_report
from src.threshold_optimizer import compute_threshold_metrics, plot_threshold_analysis, compute_business_tradeoff_table
from src.drift_detection  import simulate_drift, run_drift_monitoring
from src.expected_loss    import generate_portfolio_risk_report
from src.fairness         import run_fairness_audit
from sklearn.metrics      import roc_auc_score

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("outputs", "pipeline.log"), mode="w"),
    ]
)
logger = logging.getLogger(__name__)


def _step(n: int, total: int, msg: str):
    """Print a formatted step header."""
    print(f"\n{'='*60}")
    print(f"  Step {n}/{total} — {msg}")
    print(f"{'='*60}")


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Credit Risk Scoring Pipeline")
    p.add_argument("--skip-eda",    action="store_true", help="Skip EDA plots")
    p.add_argument("--skip-optuna", action="store_true", help="Use default LGB params")
    p.add_argument("--sample",      type=int, default=None,
                   help="Use only N rows (for fast testing)")
    return p.parse_args()


def main():
    args   = parse_args()
    t0     = time.time()
    STEPS  = 13

    os.makedirs(MODEL_PATH,   exist_ok=True)
    os.makedirs(PLOTS_PATH,   exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Load data
    # ──────────────────────────────────────────────────────────────────────────
    _step(1, STEPS, "Load Data")
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Dataset not found at: {RAW_DATA_PATH}")
        logger.error("Download application_train.csv from Kaggle and place it in data/raw/")
        sys.exit(1)

    df_raw = load_data(RAW_DATA_PATH)
    if args.sample:
        df_raw = df_raw.sample(n=min(args.sample, len(df_raw)),
                               random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info(f"Sampled {len(df_raw):,} rows for fast testing.")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — EDA
    # ──────────────────────────────────────────────────────────────────────────
    _step(2, STEPS, "Exploratory Data Analysis")
    if args.skip_eda:
        logger.info("--skip-eda flag set; skipping EDA plots.")
    else:
        run_full_eda(df_raw)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Feature engineering
    # ──────────────────────────────────────────────────────────────────────────
    _step(3, STEPS, "Feature Engineering")
    df_feat = engineer_all_features(df_raw.copy())

    # Keep a copy of the raw df aligned with the engineered features
    # (for fairness audit which needs original categorical columns)
    df_raw_aligned = df_raw.loc[df_feat.index].reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    _step(4, STEPS, "Preprocessing (impute -> encode -> scale -> SMOTE)")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_pipeline(df_feat)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Save feature names so Streamlit can align single-row inputs
    import json
    fn_path = os.path.join(MODEL_PATH, "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Feature names saved -> {fn_path}")


    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Train all models
    # ──────────────────────────────────────────────────────────────────────────
    _step(5, STEPS, "Model Training")
    results = run_full_training_pipeline(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        skip_optuna=args.skip_optuna,
    )
    best_model = results["LightGBM"][0]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Calibrate
    # ──────────────────────────────────────────────────────────────────────────
    _step(6, STEPS, "Probability Calibration")
    cal_model = calibrate_model(best_model, X_train, y_train)
    compare_calibration(best_model, cal_model, X_test, y_test)
    test_probs = get_calibrated_probabilities(cal_model, X_test)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 7 — Evaluate
    # ──────────────────────────────────────────────────────────────────────────
    _step(7, STEPS, "Evaluation & Plots")
    # Build models_dict for ROC comparison (use calibrated probs wrapper)
    eval_models = {
        name: (m, auc) for name, (m, auc) in results.items()
    }
    generate_full_evaluation_report(
        cal_model, X_test, y_test,
        feature_names=feature_names,
        models_dict=eval_models,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 8 — Scorecard
    # ──────────────────────────────────────────────────────────────────────────
    _step(8, STEPS, "FICO-Style Scorecard")
    # Build a result df aligned with test set
    test_idx = df_feat.index[-len(y_test):]   # approximate alignment
    df_scored = pd.DataFrame({"TARGET": y_test})
    df_scored = add_scores_to_dataframe(df_scored, test_probs)
    plot_score_distribution(df_scored["CREDIT_SCORE"].values, y_test)
    generate_scorecard_report(df_scored)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 9 — SHAP
    # ──────────────────────────────────────────────────────────────────────────
    _step(9, STEPS, "SHAP Explainability")
    try:
        explainer, shap_vals, X_sample = compute_shap_values(
            best_model, X_train, X_test, sample_size=500
        )
        plot_shap_summary(shap_vals, X_sample, feature_names)
        plot_shap_bar(shap_vals, X_sample, feature_names)
        plot_shap_waterfall(explainer, X_sample, idx=0, feature_names=feature_names)
        plot_shap_force(explainer, shap_vals, X_sample, idx=0, feature_names=feature_names)
        plot_shap_dependence(shap_vals, X_sample,
                             feature="EXT_SOURCE_2",
                             interaction_feature="DEBT_TO_INCOME",
                             feature_names=feature_names)
        # Save explainer for Streamlit live scoring
        joblib.dump(explainer, os.path.join(MODEL_PATH, "explainer.pkl"))
        logger.info("SHAP explainer saved → models/explainer.pkl")
    except Exception as e:
        logger.warning(f"SHAP step failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 10 — Threshold optimisation
    # ──────────────────────────────────────────────────────────────────────────
    _step(10, STEPS, "Threshold Optimisation")
    metrics_df = compute_threshold_metrics(y_test, test_probs)
    plot_threshold_analysis(metrics_df)
    loan_amounts = df_feat.iloc[-len(y_test):]["AMT_CREDIT"].values \
        if "AMT_CREDIT" in df_feat.columns else np.ones(len(y_test)) * 500_000
    compute_business_tradeoff_table(y_test, test_probs, loan_amounts)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 11 — Expected Loss
    # ──────────────────────────────────────────────────────────────────────────
    _step(11, STEPS, "Expected Loss / Portfolio Report")
    generate_portfolio_risk_report(test_probs, loan_amounts, y_true=y_test)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 12 — Drift detection
    # ──────────────────────────────────────────────────────────────────────────
    _step(12, STEPS, "Drift Detection (PSI)")
    X_prod = simulate_drift(X_test, drift_strength=0.3, feature_names=feature_names)
    run_drift_monitoring(X_train, X_prod, feature_names=feature_names)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 13 — Fairness audit
    # ──────────────────────────────────────────────────────────────────────────
    _step(13, STEPS, "Fairness Audit")
    test_n    = len(y_test)
    df_test_raw = df_raw_aligned.iloc[-test_n:].reset_index(drop=True)

    # Add engineered AGE_YEARS if available
    if "DAYS_BIRTH" in df_test_raw.columns:
        df_test_raw["AGE_YEARS"] = np.abs(df_test_raw["DAYS_BIRTH"]) / 365.25

    y_pred_bin = (test_probs >= 0.5).astype(int)
    run_fairness_audit(df_test_raw, y_test, y_pred_bin, test_probs)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\n  All plots  → {PLOTS_PATH}/")
    print(f"  Reports    → {REPORTS_PATH}/")
    print(f"  Models     → {MODEL_PATH}/")
    print(f"\n  Launch dashboard:")
    print(f"  streamlit run app/streamlit_app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
