# Credit Risk Scoring System

A **production-grade Credit Risk Scoring System** built with LightGBM, SHAP, Optuna, and Streamlit.

---

## 📁 Project Structure

```
credit_risk_scoring/
├── data/raw/                  ← place application_train.csv here
├── notebooks/EDA.ipynb        ← exploratory analysis notebook
├── src/
│   ├── config.py              ← constants, paths, hyperparameters
│   ├── eda.py                 ← EDA functions
│   ├── preprocessing.py       ← cleaning, imputation, SMOTE
│   ├── feature_engineering.py ← domain feature creation
│   ├── train.py               ← model training + Optuna
│   ├── evaluate.py            ← metrics and plots
│   ├── calibration.py         ← isotonic probability calibration
│   ├── scorecard.py           ← FICO-style score (300–850)
│   ├── threshold_optimizer.py ← business threshold analysis
│   ├── drift_detection.py     ← PSI monitoring
│   ├── expected_loss.py       ← PD × LGD × EAD
│   ├── explainability.py      ← SHAP analysis
│   └── fairness.py            ← bias audit
├── app/streamlit_app.py       ← 5-page Streamlit dashboard
├── models/                    ← saved .pkl files
├── outputs/plots/             ← all saved figures
├── outputs/reports/           ← CSV reports
├── main.py                    ← pipeline orchestrator
├── MODEL_CARD.md              ← responsible AI documentation
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download `application_train.csv` from [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it at:
```
data/raw/application_train.csv
```

### 3. Run the full pipeline
```bash
# Full pipeline (takes ~30–60 minutes with Optuna)
python main.py

# Fast dev run (5000 rows, no EDA, no Optuna)
python main.py --sample 5000 --skip-eda --skip-optuna
```

### 4. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ CLI Flags

| Flag             | Description                                     |
|------------------|-------------------------------------------------|
| `--skip-eda`     | Skip EDA plots (saves ~2 minutes)               |
| `--skip-optuna`  | Use default LightGBM params (saves ~15 minutes) |
| `--sample N`     | Use only N rows for fast testing                |

---

## 📊 Pipeline Phases

| Step | Module                  | Description                                  |
|------|-------------------------|----------------------------------------------|
| 1    | `eda.py`                | Load data + 6 EDA visualisations             |
| 2    | `feature_engineering.py`| 10+ engineered features (ratios, ages, SHAP) |
| 3    | `preprocessing.py`      | Impute → Encode → Scale → SMOTE              |
| 4    | `train.py`              | Logistic → Random Forest → LightGBM + Optuna |
| 5    | `calibration.py`        | Isotonic calibration (CV=5)                  |
| 6    | `evaluate.py`           | ROC, PR, confusion matrix, reports           |
| 7    | `scorecard.py`          | FICO-style score 300–850 + grade             |
| 8    | `explainability.py`     | SHAP summary, waterfall, dependence          |
| 9    | `threshold_optimizer.py`| F2-optimal threshold + business tradeoffs    |
| 10   | `expected_loss.py`      | PD × LGD × EAD portfolio report             |
| 11   | `drift_detection.py`    | PSI monitoring (simulated production drift)  |
| 12   | `fairness.py`           | Gender / Education / Age group audit         |

---

## 🖥️ Streamlit Dashboard Pages

| Page               | Description                                          |
|--------------------|------------------------------------------------------|
| 💳 Risk Scorer      | Enter applicant details → get credit score & SHAP    |
| 📈 Model Performance| ROC, PR, calibration, threshold, score distribution  |
| 🔍 SHAP Explorer    | Global importance, dependence, waterfall plots       |
| 🏦 Portfolio         | EL by grade, PSI drift, business tradeoff table      |
| ⚖️ Fairness Audit   | Approval rate by gender / education / age + Model Card|

---

## 📈 Key Results (approximate)

| Model              | AUC-ROC |
|--------------------|---------|
| Logistic Regression| ~0.70   |
| Random Forest      | ~0.73   |
| LightGBM (tuned)   | ~0.77   |

---

## 📋 Key Design Decisions

- **SMOTE ratio 0.3** — avoids over-sampling while improving minority recall
- **Isotonic calibration** — ensures probabilities are reliable for EL calculations
- **F2 threshold** — credit decisions penalise missed defaults more than false alarms
- **SHAP TreeExplainer** — fast and exact for tree models; used for both audit and live explanations
- **PSI monitoring** — tracks feature distribution shift between train and production

---

## ⚖️ Regulatory Compliance

- Explainable AI (SHAP reasons for each decision)
- Fairness audit across gender, education, age
- Model Card with full documentation
- Adverse action notices supported

See [MODEL_CARD.md](MODEL_CARD.md) for full responsible AI documentation.
