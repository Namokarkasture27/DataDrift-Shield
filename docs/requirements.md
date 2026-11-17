# DataDrift-Shield — Requirements Document

Functional & Non-Functional Requirements, and Module/Sprint Plan.

This document lists what DataDrift-Shield (a self-healing ML system for data
drift detection, explainable diagnosis, and automated retraining) is required
to do, organized into modules. Each module is treated as a sprint: Modules
1–5 are complete; Module 6 ships with this review.

## 1. Functional Requirements (by module)

**Module 1 — Baseline Model (Done)**
- FR1.1: The system shall ingest the IBM Telco Customer Churn dataset and preprocess it for training.
- FR1.2: The system shall train an XGBoost classification model on the processed data.
- FR1.3: The system shall evaluate the model using accuracy and ROC-AUC (achieved: 80.5% accuracy, 84.5% ROC-AUC).

**Module 2 — Drift Detection (Done)**
- FR2.1: The system shall compare incoming data's distribution against the training data's distribution using the KS-test.
- FR2.2: The system shall flag a feature as drifted when the KS-test p-value falls below a set threshold.
- FR2.3: The system shall support a controlled drift experiment (e.g., shifting a feature by a fixed %) for testing.

**Module 3 — Explainability (Done)**
- FR3.1: The system shall compute SHAP values to identify which features are driving a prediction or performance change.
- FR3.2: The system shall rank drifted features by their contribution score.

**Module 4 — GenAI Reporting (Done)**
- FR4.1: The system shall send drift and SHAP results to the Gemini API.
- FR4.2: The system shall generate a plain-language diagnostic report explaining the drift and whether retraining is needed.

**Module 5 — Automated Retraining & Validation (Done)**
- FR5.1: The system shall automatically retrain the model on newer data once drift is detected.
- FR5.2: The system shall validate the retrained ("challenger") model against the current ("champion") model on the same test data.
- FR5.3: The system shall replace the production model only if the challenger outperforms the champion on the chosen metric.

**Module 6 — Monitoring Dashboard (Done)**
- FR6.1: The system shall display live drift status, SHAP explanations, and the Gemini report on a Streamlit dashboard.
- FR6.2: The system shall show model version history and retraining logs.

## 2. Non-Functional Requirements

- Performance: Drift detection and SHAP computation shall complete within a few seconds for a standard data batch.
- Reliability: The system shall never auto-replace the production model unless validation (Module 5) passes.
- Usability: Gemini-generated reports must be understandable to a non-technical reader, not just raw statistics.
- Portability: The system shall run on a standard laptop, with no GPU, using only free/open-source libraries.
- Maintainability: Each stage (detection, explainability, reporting, retraining) shall be an independently testable module.
- Security: API keys (Gemini) shall be stored as environment variables, never hardcoded in the source code.

## 3. Use Case Summary

**Actor:** ML Engineer / Admin (the only human actor; monitors the system and reviews its decisions).

**Use cases, in sequence:** Detect Drift → Explain Drift (SHAP) → Generate Report (Gemini) → Retrain Model → Validate & Replace Model → View Dashboard. The Admin starts monitoring; the system then runs detection, explanation, and reporting automatically, and only retrains/replaces the model after passing validation. All results are visible to the Admin on the dashboard.
