# DataDrift-Shield: Requirements Document

Functional and Non-Functional Requirements, Use Cases and Sprint Plan.

Team No. 13, School of Computing-CSE, MIT Vishwaprayag University.
Word version of this document: `docs/DataDrift-Shield-Requirements.docx`

## 1. Introduction

DataDrift-Shield watches a deployed AI model. It finds when new data is different from the data the model learned from (data drift), explains the change in simple words, and trains a new model. The old model is replaced only if the new one is better.

- **User:** one person, the ML Engineer (Admin), who runs the checks and reads the results.
- **External system:** the Gemini API, which writes the report.
- **Data:** IBM Telco Customer Churn sample data (a fictional telecom company, 7,043 customers, 21 columns).

**Priority levels:** 1 = essential and required, 2 = desirable, 3 = extra feature.

## 2. Functional Requirements (what the system must do)

| ID | Requirement | Priority | Status |
|---|---|---|---|
| **Module 1: Baseline Model** (`src/train_baseline.py`) | | | |
| FR1.1 | The system shall load the IBM Telco Customer Churn dataset and prepare it for training. | 1 | Done |
| FR1.2 | The system shall train an XGBoost model that predicts if a customer will leave (churn). | 1 | Done |
| FR1.3 | The system shall measure the model with accuracy and ROC-AUC (result: 80.1% and 0.843). | 1 | Done |
| **Module 2: Drift Detection** (`src/detect_drift.py`) | | | |
| FR2.1 | The system shall compare new data with training data, column by column, using the KS-test. | 1 | Done |
| FR2.2 | The system shall mark a column as drifted when its p-value is below 0.05. | 1 | Done |
| FR2.3 | The system shall let us shift a column by a fixed % (for example MonthlyCharges +20%) to test drift. | 2 | Done |
| **Module 3: Explainability** (`src/explain_model.py`) | | | |
| FR3.1 | The system shall use SHAP to show which columns have the most effect on the model's answers. | 1 | Done |
| FR3.2 | The system shall rank the drifted columns by how much they matter to the model. | 2 | Done |
| **Module 4: AI Report** (`src/generate_report.py`) | | | |
| FR4.1 | The system shall send the drift and SHAP results to the Gemini API. | 1 | Done |
| FR4.2 | The system shall save a simple-words report that says what changed, why it matters, and if retraining is needed. | 1 | Done |
| **Module 5: Retrain and Check** (`src/retrain_validate.py`) | | | |
| FR5.1 | The system shall train a new (challenger) model on newer data when drift is found. | 1 | Done |
| FR5.2 | The system shall test the challenger and the current (champion) model on the same test data. | 1 | Done |
| FR5.3 | The system shall replace the champion only if the challenger scores higher, and shall log every run. | 1 | Done |
| **Module 6: Dashboard** (`app.py`) | | | |
| FR6.1 | The system shall show drift status, SHAP plots and the Gemini report on one Streamlit screen. | 2 | Done |
| FR6.2 | The system shall show the retraining log (champion vs challenger, promoted or not). | 2 | Done |
| **Future work** | | | |
| FR7.1 | The system shall read live or streaming data instead of a saved batch. | 3 | Planned |

## 3. Non-Functional Requirements (how well the system must work)

| ID | Property | Requirement (with a measure) | Type | How we check |
|---|---|---|---|---|
| NFR1 | Performance | Drift test and SHAP shall finish in under 30 seconds for the 7,043-row dataset on a normal laptop. | Product | Time the scripts |
| NFR2 | Reliability | The production model shall never be replaced unless the challenger has a higher ROC-AUC on the same test data. | Product | Check the `promoted` column in `outputs/retraining_log.csv` |
| NFR3 | Accuracy of drift test | A planned +20% shift shall be flagged (p < 0.05). Unchanged columns shall not be flagged. | Product | Run `detect_drift.py` |
| NFR4 | Usability | Each Gemini report shall use plain words and cover: what changed, why it matters, and the decision. A non-technical reader shall understand it. | Product | Reviewer reads the report |
| NFR5 | Portability | The system shall run on a normal laptop with no GPU, using only free and open-source libraries (`requirements.txt`). | Product | Fresh install and run |
| NFR6 | Maintainability | Each stage shall be its own script and shall be testable alone. Unit tests shall pass with pytest. | Organisational | Run `pytest tests/` |
| NFR7 | Security | The Gemini API key shall be kept in a `.env` file that is listed in `.gitignore`. It shall never be committed to GitHub. | Organisational | Check the repository |
| NFR8 | Version control | Code shall be on GitHub with regular commits, a clear folder structure and a README. | Organisational | Check commit history |
| NFR9 | Data and privacy | Only IBM's public sample dataset (a fictional telecom company) shall be used. No real customer data. | External | Dataset source |

## 4. Use Cases

**Actors:** ML Engineer (Admin), Gemini API (external).

| UC | Use case | Actor | What happens | Requirements |
|---|---|---|---|---|
| UC1 | Detect drift | Admin | Compare new data with training data using the KS-test. Flag columns with p < 0.05. | FR2.1-FR2.3 |
| UC2 | Explain drift | System | Use SHAP to show which columns matter most and rank the drifted ones. | FR3.1-FR3.2 |
| UC3 | Generate report | Gemini API | Turn the results into a simple-words report with a retraining suggestion. | FR4.1-FR4.2 |
| UC4 | Retrain challenger model | System | Train a new model on newer data. | FR5.1 |
| UC5 | Validate and replace model | System | Test both models on the same data. Replace the old one only if the new one is better. Log the result. | FR5.2-FR5.3 |
| UC6 | View dashboard | Admin | See drift status, SHAP plots, report and retraining log on one screen. | FR6.1-FR6.2 |

Flow: the Admin starts UC1. UC1 includes UC2, which includes UC3, UC4 and UC5 in order. UC6 shows the results of all steps.

## 5. Sprint Plan

Each sprint takes a small set of requirements through requirements, design, coding and testing before the next one starts.

| Sprint | Weeks | Modules | Requirements | Status |
|---|---|---|---|---|
| Sprint 1 | Weeks 3-4 | M1 Baseline model, M2 Drift detection | FR1.1-FR1.3, FR2.1-FR2.3 | Done |
| Sprint 2 | Weeks 5-8 | M3 SHAP, M4 Gemini report | FR3.1-FR3.2, FR4.1-FR4.2 | Done |
| Sprint 3 | Weeks 9-10 | M5 Retrain and check | FR5.1-FR5.3 | Done |
| Sprint 4 | Weeks 11-12 | M6 Streamlit dashboard | FR6.1-FR6.2 | Done |

**Traceability:** each module maps to one code file (shown in the Section 2 headings), so every requirement can be found in the code.
