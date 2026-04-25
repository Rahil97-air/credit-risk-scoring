# Credit Risk Scoring System — Model Card
<!-- Model Card version 1.0 — April 2026 -->

---

## 1. Model Details

| Field            | Value                                                   |
|------------------|---------------------------------------------------------|
| **Name**         | Credit Risk Scoring Model v1.0                          |
| **Version**      | 1.0.0                                                   |
| **Type**         | Gradient Boosted Tree (LightGBM) + Isotonic Calibration |
| **Date**         | April 2026                                              |
| **Framework**    | LightGBM 4.x, scikit-learn 1.3+                        |
| **Output**       | Default probability (PD), FICO-style score (300–850)    |
| **Maintainer**   | Credit Risk ML Team                                     |

---

## 2. Intended Use

**Primary use:** Assessing probability of default (PD) for personal loan applicants in retail banking.

**Intended users:** Credit analysts, risk officers, automated loan decisioning systems.

**Use cases:**
- Pre-screening loan applications
- Setting interest rate tiers
- Portfolio-level expected loss calculation
- Regulatory capital estimation (Basel II/III PD component)

---

## 3. Out-of-Scope Uses

> [!WARNING]
> The following uses are **not supported** and may produce unreliable results:

- **Mortgage loans** — different collateral structure and LGD profile
- **Business / SME loans** — requires different feature set (financial statements, trade credit)
- **Credit card credit lines** — revolving exposure requires separate EAD treatment
- **Jurisdictions outside India** — model was trained on Indian domestic loan data
- **Real-time fraud detection** — model is not designed for fraud scoring

---

## 4. Training Data

| Field             | Value                                                    |
|-------------------|----------------------------------------------------------|
| **Source**        | Home Credit Default Risk (Kaggle public dataset)         |
| **File**          | `application_train.csv`                                  |
| **Size**          | ~307,511 rows, 122 raw features                          |
| **Date Range**    | Historical lending data (exact dates not disclosed)      |
| **Target**        | `TARGET`: 1 = loan default, 0 = repaid                   |
| **Class Balance** | ~8.1% default (minority class)                           |
| **SMOTE**         | Applied at ratio 0.3 to minority class in training split |

---

## 5. Evaluation Data

| Field             | Value                                  |
|-------------------|----------------------------------------|
| **Split method**  | Stratified train/test split (80/20)    |
| **Test size**     | ~61,500 rows                           |
| **Stratified on** | `TARGET` column                        |

---

## 6. Performance Metrics

| Metric                  | Value (approx.) |
|-------------------------|-----------------|
| AUC-ROC                 | ~0.77           |
| Average Precision (AP)  | ~0.31           |
| F2-Score (β=2)          | ~0.28           |
| Calibration Error (ECE) | ~0.03           |

> [!NOTE]
> Exact values depend on the Optuna-tuned hyperparameters. Run `main.py` to regenerate with your dataset split.

---

## 7. Known Limitations

1. **Reject Inference Bias:** The dataset only contains accepted applications. Rejected applicants (who may have higher default rates) are absent, leading to potential underestimation of risk at score thresholds.

2. **Geographic Scope:** All borrowers are from India. Model may not generalise to other markets.

3. **Temporal Drift:** Credit behaviour changes over time. The model should be retrained at least annually and monitored monthly using PSI.

4. **Self-Reported Income:** `AMT_INCOME_TOTAL` is borrower-declared and not verified in this dataset.

5. **External Scores Opacity:** `EXT_SOURCE_1/2/3` are third-party bureau scores with no documentation on their construction.

---

## 8. Fairness Considerations

The model is audited for disparate impact across the following sensitive attributes:

| Attribute            | Metric Checked       | Disparity Threshold |
|----------------------|----------------------|---------------------|
| `CODE_GENDER`        | Approval rate        | >10% gap flagged    |
| `NAME_EDUCATION_TYPE`| Approval rate        | >10% gap flagged    |
| Age group (<30, 30-45, 45-60, 60+) | Approval rate | >10% gap flagged |

**Mitigation steps taken:**
- `class_weight="balanced"` used in all classifiers
- Threshold optimization uses F2-score (prioritises recall of defaults over precision)
- Fairness audit run post-training; results saved to `outputs/reports/fairness_audit.csv`

> [!CAUTION]
> This model must **not** be used as the sole basis for credit decisions. Human review is required for borderline cases (score 600–700). Denial reasons must be provided to applicants per ECOA/RBI guidelines.

---

## 9. Regulatory Notes

- **RBI Master Direction on Digital Lending (2022):** Lenders must ensure algorithmic credit decisions are explainable. SHAP explanations are integrated for this purpose.
- **Equal Credit Opportunity Act (ECOA) analogue:** Adverse action notices with top denial reasons (SHAP-based) are surfaced by the system.
- **Basel II/III:** PD output is suitable for IRB (Internal Ratings-Based) approach input, subject to bank-specific validation and regulatory approval.
- **GDPR / Data Privacy:** No personal identifiers (name, PAN, Aadhaar) are used as model features.

---

## 10. Recommendations for Use

1. **Threshold:** Use the F2-optimal threshold (prioritises catching defaults) for portfolio management; use a higher threshold (e.g. 0.5) for revenue maximisation.
2. **Monitoring:** Run PSI drift monitoring monthly on incoming applications.
3. **Recalibration:** Recalibrate isotonic model quarterly or after any significant policy change.
4. **Human oversight:** All REVIEW-band applicants (score 600–700) should receive manual underwriter review.
5. **Adverse action:** Provide top-5 SHAP reasons for any rejection to comply with explainability requirements.

---

## 11. Contact / Maintainer

| Field       | Value                           |
|-------------|---------------------------------|
| Team        | Credit Risk ML Team             |
| Repository  | `credit_risk_scoring/`          |
| Issues      | Raise a ticket in your internal issue tracker |
| Version log | See `CHANGELOG.md` (if present) |
