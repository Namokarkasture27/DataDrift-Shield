# DataDrift-Shield — Architecture & Design

## 1. Architecture (updated for Review-2)

```
                 ┌─────────────────────┐
                 │   Telco Churn CSV    │
                 │   (data/*.csv)       │
                 └──────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ 1. train_baseline.py  │  --> models/baseline_xgboost.joblib
                 │   (XGBoost classifier)│      (the "champion" model)
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ 2. detect_drift.py    │  KS-test: reference batch vs.
                 │   (KS-test)           │  incoming batch, per feature
                 └───────────┬───────────┘
                             │ drift evidence
                 ┌───────────▼───────────┐
                 │ 3. explain_model.py   │  --> reports/shap_*.png
                 │   (SHAP)              │  which features matter most
                 └───────────┬───────────┘
                             │ drift evidence + SHAP ranking
                 ┌───────────▼───────────┐
                 │ 4. generate_report.py │  --> reports/gemini_diagnostic_report.md
                 │   (Gemini API)        │  plain-language diagnosis
                 └───────────┬───────────┘
                             │ recommendation
                 ┌───────────▼───────────┐
                 │ 5. retrain_validate.py│  trains a "challenger" model,
                 │   (champion/challenger│  compares it to the champion on
                 │    validation gate)   │  the SAME test set (ROC-AUC)
                 └───────────┬───────────┘
                             │ promote only if challenger wins
                 ┌───────────▼───────────┐
                 │ models/ (champion)    │  outputs/retraining_log.csv
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │ 6. app.py (Streamlit) │  drift status, SHAP plots,
                 │   Monitoring Dashboard│  Gemini report, version history
                 └────────────────────────┘
```

**Key design rule:** the production model is never replaced automatically just
because drift was detected. Drift only triggers retraining; the retrained
("challenger") model is validated against the current ("champion") model on
identical held-out data, and only promoted if it measurably performs better.
This is the validation gate referenced in the requirements document
(`docs/requirements.md` / FR5.2–FR5.3).

## 2. Requirements and mapping

See `docs/requirements.md` for the full Functional and Non-Functional
Requirements, grouped by module, and a short use-case summary. Each module
above maps 1:1 to a requirements block (Module 1 -> FR1.x, etc.), and to a
script in `src/`.

## 3. Why XGBoost

- Tabular, mixed numeric/categorical data (the Telco dataset) is exactly what
  gradient-boosted trees handle well, without needing feature scaling.
- Fast to retrain, which matters for Module 5 (the system needs to retrain
  repeatedly, not just once).
- SHAP's `TreeExplainer` is exact and fast for tree models, which is what
  Module 3 (explainability) relies on.

## 4. Flowchart — drift-to-retrain decision

```
          ┌───────────────┐
          │ New data batch │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ Run KS-test per │
          │ feature vs. ref │
          └───────┬────────┘
                  │
         ┌────────▼─────────┐
         │ p-value < 0.05 on │  No  ───►  No action; keep monitoring
         │ any feature?      │
         └────────┬──────────┘
                  │ Yes
          ┌───────▼────────┐
          │ Run SHAP to see  │
          │ which features   │
          │ drive the model  │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ Gemini generates │
          │ plain-language   │
          │ diagnostic report│
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ Retrain challenger│
          │ on newer data     │
          └───────┬────────┘
                  │
         ┌────────▼─────────┐
         │ Challenger ROC-AUC │  No  ───►  Keep champion in production
         │ > champion ROC-AUC?│
         └────────┬──────────┘
                  │ Yes
          ┌───────▼────────┐
          │ Promote challenger│
          │ to production      │
          │ (becomes champion) │
          └───────┬────────┘
                  │
          ┌───────▼────────┐
          │ Log the run in   │
          │ retraining_log.csv│
          └────────────────┘
```

## 5. Database design

No database is used. All intermediate state (the trained model, SHAP plots,
the Gemini report, the retraining log) is stored as flat files under
`models/`, `reports/`, and `outputs/`, which is sufficient for a single-user
monitoring dashboard at this project's scale.
