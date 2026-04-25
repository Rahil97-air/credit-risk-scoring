"""
streamlit_app.py — 5-page Credit Risk Scoring Dashboard.
Run: streamlit run app/streamlit_app.py  (from credit_risk_scoring/ root)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import json
from src.config import MODEL_PATH, PLOTS_PATH, REPORTS_PATH
from src.scorecard import probability_to_score, score_to_grade
from src.feature_engineering import engineer_all_features
from src.preprocessing import impute_features, encode_categoricals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Scoring System",
                   page_icon="💳", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.metric-card{background:linear-gradient(135deg,#1a1f2e,#252b3b);border:1px solid #2d3748;
  border-radius:12px;padding:20px 24px;margin:6px 0;box-shadow:0 4px 15px rgba(0,0,0,.3);}
.metric-card h3{color:#a0aec0;font-size:13px;font-weight:600;letter-spacing:.05em;
  margin:0 0 6px;text-transform:uppercase;}
.metric-card .val{font-size:32px;font-weight:700;margin:0;}
.section-header{font-size:20px;font-weight:700;color:#e2e8f0;
  border-left:4px solid #667eea;padding-left:12px;margin:24px 0 12px;}
.disp-warn{background:rgba(231,76,60,.15);border:1px solid #e74c3c;border-radius:8px;
  padding:10px 16px;color:#e74c3c;font-weight:600;margin:8px 0;}
.chip-a{background:#1a4731;color:#2ecc71;border:1px solid #2ecc71;
  padding:8px 20px;border-radius:20px;font-weight:700;font-size:16px;display:inline-block;}
.chip-r{background:#3d2e0a;color:#f39c12;border:1px solid #f39c12;
  padding:8px 20px;border-radius:20px;font-weight:700;font-size:16px;display:inline-block;}
.chip-x{background:#3d0a0a;color:#e74c3c;border:1px solid #e74c3c;
  padding:8px 20px;border-radius:20px;font-weight:700;font-size:16px;display:inline-block;}
.block-container{padding-top:1.5rem;}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    for fname in ("calibrated_model.pkl", "lgbm_model.pkl"):
        p = os.path.join(MODEL_PATH, fname)
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

@st.cache_resource(show_spinner="Loading scaler…")
def load_scaler():
    p = os.path.join(MODEL_PATH, "scaler.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_feature_names():
    p = os.path.join(MODEL_PATH, "feature_names.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def show_img(fname, caption=""):
    p = os.path.join(PLOTS_PATH, fname)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.info(f"📊 `{fname}` not yet generated — run `main.py` first.")

def metric_card(title, value, color):
    st.markdown(f'<div class="metric-card"><h3>{title}</h3>'
                f'<p class="val" style="color:{color}">{value}</p></div>',
                unsafe_allow_html=True)


# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = ["💳 Risk Scorer","📈 Model Performance","🔍 SHAP Explorer",
         "🏦 Portfolio Dashboard","⚖️ Fairness Audit"]

with st.sidebar:
    st.markdown("## 💳 Credit Risk System")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Run `python main.py` to generate all plots & reports.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Risk Scorer
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.markdown('<div class="section-header">Single Applicant Risk Assessment</div>',
                unsafe_allow_html=True)
    try:
        model, mpath = load_model()
        scaler = load_scaler()
        if model is None:
            st.warning("No model found — run `python main.py` first.")
            st.stop()

        st.sidebar.markdown("### 📋 Applicant Details")
        amt_income  = st.sidebar.number_input("Annual Income (₹)",   0, 10_000_000, 300_000, 10_000)
        amt_credit  = st.sidebar.number_input("Loan Amount (₹)",     0, 10_000_000, 500_000, 10_000)
        amt_annuity = st.sidebar.number_input("Annual Annuity (₹)",  0,  1_000_000,  30_000,  1_000)
        amt_goods   = st.sidebar.number_input("Goods Price (₹)",     0,  5_000_000, 450_000, 10_000)
        cnt_fam     = st.sidebar.number_input("Family Members", 1, 20, 2)
        age_yrs     = st.sidebar.slider("Age (years)", 18, 70, 35)
        days_emp    = st.sidebar.number_input("Days Employed (negative)", -10000, 0, -1000)
        ext1 = st.sidebar.slider("Ext. Score 1", 0.0, 1.0, 0.50, 0.01)
        ext2 = st.sidebar.slider("Ext. Score 2", 0.0, 1.0, 0.50, 0.01)
        ext3 = st.sidebar.slider("Ext. Score 3", 0.0, 1.0, 0.50, 0.01)
        edu  = st.sidebar.selectbox("Education", ["Higher education",
               "Secondary / secondary special","Incomplete higher",
               "Lower secondary","Academic degree"])
        contract = st.sidebar.selectbox("Contract Type", ["Cash loans","Revolving loans"])
        gender   = st.sidebar.selectbox("Gender", ["M","F","XNA"])

        st.markdown("---")
        
        row = pd.DataFrame([{
            "AMT_INCOME_TOTAL": amt_income, "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity, "AMT_GOODS_PRICE": amt_goods,
            "CNT_FAM_MEMBERS": cnt_fam, "DAYS_BIRTH": -int(age_yrs*365.25),
            "DAYS_EMPLOYED": days_emp, "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
            "NAME_EDUCATION_TYPE": edu, "NAME_CONTRACT_TYPE": contract,
            "CODE_GENDER": gender, "TARGET": 0,
        }])
        # Feature engineering
        row = engineer_all_features(row).drop(columns=["TARGET"], errors="ignore")
        # Apply same imputation + encoding as training pipeline
        row = impute_features(row)
        row = encode_categoricals(row)
        # Align columns with what the scaler expects
        feature_names = load_feature_names()
        if feature_names is not None:
            for c in feature_names:
                if c not in row.columns:
                    row[c] = 0
            row = row[feature_names]
        else:
            row = row.select_dtypes(include=np.number).fillna(0)
        X = scaler.transform(row.values) if scaler is not None else row.values

        prob  = float(model.predict_proba(X)[0, 1])
        score = probability_to_score(prob)
        grade = score_to_grade(score)

        if score >= 700:   color, chip, decision = "#2ecc71", "chip-a", "APPROVE"
        elif score >= 600: color, chip, decision = "#f39c12", "chip-r", "REVIEW"
        else:              color, chip, decision = "#e74c3c", "chip-x", "REJECT"

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Credit Score",       str(score),         color)
        with c2: metric_card("Default Probability", f"{prob*100:.1f}%", color)
        with c3: metric_card("Credit Grade",        grade,              color)

        st.markdown("---")
        st.markdown(f'<span class="{chip}">Decision: {decision}</span>',
                    unsafe_allow_html=True)
        st.markdown("**Risk Level**")
        st.progress(min(prob, 1.0))

        # SHAP top reasons
        st.markdown('<div class="section-header">Top Risk Factors</div>',
                    unsafe_allow_html=True)
        exp_path = os.path.join(MODEL_PATH, "explainer.pkl")
        if os.path.exists(exp_path):
            try:
                explainer = joblib.load(exp_path)
                from src.explainability import get_top_reasons
                reasons = get_top_reasons(explainer, X, list(row.columns), n=5)
                rdf = pd.DataFrame(reasons)
                rdf["impact"] = rdf["shap_value"].apply(
                    lambda v: f"▲ {v:.4f}" if v>0 else f"▼ {v:.4f}")
                rdf["icon"] = rdf["direction"].apply(
                    lambda d: "🔴" if "increases" in d else "🟢")
                st.dataframe(rdf[["icon","feature","value","impact","direction"]]
                             .rename(columns={"icon":"","feature":"Feature",
                                              "value":"Value","impact":"SHAP","direction":"Direction"}),
                             use_container_width=True)
            except Exception as ex:
                st.info(f"SHAP explainer error: {ex}")
        else:
            st.info("SHAP explainer not found — run `main.py` to generate.")
    except Exception as e:
        st.error(f"Risk Scorer error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.markdown('<div class="section-header">Model Performance Dashboard</div>',
                unsafe_allow_html=True)
    try:
        eval_df = load_csv(os.path.join(REPORTS_PATH, "evaluation_report.csv"))
        if not eval_df.empty:
            st.markdown("#### 📊 Evaluation Metrics")
            st.dataframe(eval_df, use_container_width=True)
        else:
            st.info("Run `main.py` to generate evaluation_report.csv")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC Curves**"); show_img("roc_curves.png")
        with c2:
            st.markdown("**Precision-Recall Curve**"); show_img("pr_curve.png")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion Matrix**"); show_img("confusion_matrix.png")
        with c2:
            st.markdown("**Calibration Comparison**"); show_img("calibration_comparison.png")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Threshold Analysis**"); show_img("threshold_analysis.png")
        with c2:
            st.markdown("**Score Distribution**"); show_img("score_distribution.png")
    except Exception as e:
        st.error(f"Performance page error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.markdown('<div class="section-header">SHAP Explainability Explorer</div>',
                unsafe_allow_html=True)
    try:
        st.markdown("> **SHAP** measures each feature's contribution. "
                    "Positive = increases default risk, negative = decreases it.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Global Importance (bar)**")
            st.caption("Mean |SHAP| — overall feature importance.")
            show_img("shap_bar.png")
        with c2:
            st.markdown("**Impact Summary (beeswarm)**")
            st.caption("Each dot = one sample. Red = high value, blue = low.")
            show_img("shap_summary.png")
        st.markdown("---")

        st.markdown("**Feature Dependence Plot**")
        dep_files = [f for f in os.listdir(PLOTS_PATH)
                     if f.startswith("shap_dependence_") and f.endswith(".png")] \
                    if os.path.exists(PLOTS_PATH) else []
        if dep_files:
            names = [f.replace("shap_dependence_","").replace(".png","") for f in dep_files]
            chosen = st.selectbox("Select feature:", names)
            show_img(f"shap_dependence_{chosen}.png", caption=f"SHAP dependence: {chosen}")
            st.info("Each point = one applicant. Y = how much this feature pushed towards default.")
        else:
            st.info("Dependence plots not generated — run `main.py` first.")

        st.markdown("---")
        st.markdown("**Waterfall — Single Prediction**")
        wf_files = [f for f in os.listdir(PLOTS_PATH)
                    if f.startswith("shap_waterfall_") and f.endswith(".png")] \
                   if os.path.exists(PLOTS_PATH) else []
        if wf_files:
            show_img(wf_files[0], caption="Waterfall for applicant index 0")
            st.info("Shows how features moved the score from the base rate to the final prediction.")
        else:
            st.info("Waterfall plot not generated — run `main.py` first.")
    except Exception as e:
        st.error(f"SHAP Explorer error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Portfolio Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.markdown('<div class="section-header">Portfolio Risk Dashboard</div>',
                unsafe_allow_html=True)
    try:
        portfolio_df = load_csv(os.path.join(REPORTS_PATH, "portfolio_risk_report.csv"))
        tradeoff_df  = load_csv(os.path.join(REPORTS_PATH, "business_tradeoff.csv"))
        drift_df     = load_csv(os.path.join(REPORTS_PATH, "drift_report.csv"))

        if not portfolio_df.empty:
            total_el    = portfolio_df.get("total_EL",    pd.Series([0])).sum()
            avg_pd      = portfolio_df.get("avg_PD",      pd.Series([0])).mean()
            total_loans = portfolio_df.get("count",       pd.Series([0])).sum()
            hr_pct      = (portfolio_df.loc[
                portfolio_df.index.isin(["D — Poor","E — High Risk"]), "count"].sum()
                / max(total_loans, 1) * 100) if "count" in portfolio_df.columns else 0

            c1,c2,c3,c4 = st.columns(4)
            with c1: metric_card("Total Expected Loss",  f"₹{total_el:,.0f}", "#e74c3c")
            with c2: metric_card("Average PD",            f"{avg_pd*100:.2f}%", "#f39c12")
            with c3: metric_card("High-Risk Loans (D+E)", f"{hr_pct:.1f}%",     "#9b59b6")
            with c4: metric_card("Portfolio Size",         f"{int(total_loans):,}", "#2ecc71")
            st.markdown("---")

            if "total_EL" in portfolio_df.columns:
                st.markdown("**Expected Loss by Grade**")
                fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
                cmap = {"A — Excellent":"#2ecc71","B — Good":"#27ae60",
                        "C — Fair":"#f39c12","D — Poor":"#e67e22","E — High Risk":"#e74c3c"}
                ax.bar(portfolio_df.index,
                       portfolio_df["total_EL"],
                       color=[cmap.get(g,"#999") for g in portfolio_df.index],
                       edgecolor="white")
                ax.set_ylabel("Expected Loss (₹)")
                ax.set_title("Expected Loss by Credit Grade", fontweight="bold")
                ax.tick_params(axis="x", rotation=20)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            st.markdown("**Portfolio Summary**")
            st.dataframe(portfolio_df, use_container_width=True)
        else:
            st.info("Portfolio report not found — run `main.py` first.")

        st.markdown("---")
        if not tradeoff_df.empty:
            st.markdown("**Business Tradeoff at Different Thresholds**")
            st.dataframe(tradeoff_df, use_container_width=True)

        st.markdown("---")
        st.markdown("**Drift Monitoring (PSI)**")
        show_img("psi_report.png")
        if not drift_df.empty:
            vc = drift_df["status"].value_counts()
            c1,c2,c3 = st.columns(3)
            for col, s, clr in [(c1,"Stable","#2ecc71"),(c2,"Warning","#f39c12"),(c3,"Critical","#e74c3c")]:
                with col: metric_card(f"{s} Features", str(vc.get(s,0)), clr)
    except Exception as e:
        st.error(f"Portfolio Dashboard error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Fairness Audit
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.markdown('<div class="section-header">Fairness & Bias Audit</div>',
                unsafe_allow_html=True)
    try:
        fairness_df = load_csv(os.path.join(REPORTS_PATH, "fairness_audit.csv"))
        if not fairness_df.empty:
            for feat in fairness_df["sensitive_feature"].unique():
                st.markdown(f"#### {feat}")
                grp = fairness_df[fairness_df["sensitive_feature"] == feat]
                mx, mn = grp["approval_rate"].max(), grp["approval_rate"].min()
                if (mx - mn) > 0.10:
                    st.markdown(
                        f'<div class="disp-warn">⚠️ Approval rate disparity: '
                        f'{(mx-mn)*100:.1f}% gap between groups.</div>',
                        unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8,3), dpi=120)
                avg = grp["approval_rate"].mean()
                ax.bar(grp["group"].astype(str), grp["approval_rate"]*100,
                       color=["#e74c3c" if v<avg*0.9 else "#2ecc71"
                              for v in grp["approval_rate"]],
                       edgecolor="white", width=0.55)
                ax.axhline(avg*100, color="gray", linestyle="--",
                           linewidth=1.2, label=f"Avg ({avg*100:.1f}%)")
                ax.set_ylabel("Approval Rate (%)")
                ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=20)
                plt.tight_layout()
                st.pyplot(fig); plt.close()
                st.dataframe(grp.drop(columns=["sensitive_feature"]),
                             use_container_width=True)
                st.markdown("---")
        else:
            st.info("Fairness audit not found — run `main.py` first.")

        st.markdown('<div class="section-header">Model Card</div>', unsafe_allow_html=True)
        mc = os.path.join(os.path.dirname(__file__), "..", "MODEL_CARD.md")
        if os.path.exists(mc):
            st.markdown(open(mc, encoding="utf-8").read())
        else:
            st.info("MODEL_CARD.md not found.")
    except Exception as e:
        st.error(f"Fairness Audit error: {e}")
