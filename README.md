# DataDrift-Shield

A self-healing machine learning system for customer churn prediction. It trains a
baseline model, detects when incoming data has drifted away from the training
data, explains *why* using SHAP, and uses Google's Gemini API to turn that
evidence into a plain-language diagnostic report. It then decides whether the
model should be retrained and automatically replaces the production model only
if the retrained version is actually better.

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns).

## Pipeline

```
train_baseline.py  -->  detect_drift.py  -->  explain_model.py  -->  generate_report.py  -->  retrain_validate.py
   (Module 1)           (Module 2)            (Module 3)             (Module 4)               (Module 5)
```

| Module | Script | What it does | Status |
|---|---|---|---|
| 1. Baseline model | `src/train_baseline.py` | Trains an XGBoost classifier on the Telco churn dataset. | Done |
| 2. Drift detection | `src/detect_drift.py` | Runs a Kolmogorov–Smirnov test comparing a reference batch against an incoming batch; includes a controlled experiment (MonthlyCharges shifted +20%). | Done |
| 3. Explainability | `src/explain_model.py` | Computes SHAP values and saves feature-importance and summary plots. | Done |
| 4. GenAI reporting | `src/generate_report.py` | Sends the drift + SHAP evidence to Gemini, which writes a plain-language diagnostic report. | Done |
| 5. Retraining & validation | `src/retrain_validate.py` | Retrains a "challenger" model on newer data and only promotes it if it beats the current "champion" on the same test set. | Done |
| 6. Monitoring dashboard | `app.py` (Streamlit) | Shows drift status, SHAP plots, the Gemini report, and model version history in one dashboard. | Done |

## Results (baseline)

- Accuracy: **80.5%**
- ROC-AUC: **84.5%**

## Project structure

```
DataDrift-Shield/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                     # Streamlit dashboard (Module 6)
├── data/
│   └── Telco-Customer-Churn.csv   # not committed (see .gitignore); see Setup
├── src/
│   ├── train_baseline.py      # Module 1
│   ├── detect_drift.py        # Module 2
│   ├── explain_model.py       # Module 3
│   ├── generate_report.py     # Module 4
│   └── retrain_validate.py    # Module 5
├── tests/
│   └── test_drift_and_model.py
├── docs/
│   └── design.md              # architecture + requirements mapping
├── reports/                   # SHAP plots + Gemini diagnostic report (generated)
├── models/                    # saved model files (generated, not committed)
└── outputs/                   # retraining/validation logs (generated)
```

## Setup

```bash
pip install -r requirements.txt
```

Place `Telco-Customer-Churn.csv` in `data/` (not committed to keep the repo light).
For the GenAI report, create a `.env` file in the project root with:

```
GEMINI_API_KEY=your_key_here
```

## Running it

```bash
python src/train_baseline.py        # trains and saves the baseline model
python src/detect_drift.py          # runs the KS-test drift experiment
python src/explain_model.py         # SHAP plots -> reports/
python src/generate_report.py       # Gemini diagnostic report -> reports/
python src/retrain_validate.py      # retrain, validate, and promote if better
streamlit run app.py                # monitoring dashboard
```

## Tests

```bash
pytest tests/
```

## Requirements document

Full functional/non-functional requirements and the module/sprint plan: see
`docs/requirements.md` (also submitted separately as a Word document to the guide).
