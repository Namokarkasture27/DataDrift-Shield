
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import shap
from dotenv import load_dotenv
from google import genai
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your root .env file."
    )

DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_xgboost.joblib"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Load data and trained model
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

features = ["tenure", "MonthlyCharges", "TotalCharges"]
clean = df[features].dropna()

# Create reproducible reference and incoming batches
reference = clean.sample(frac=0.70, random_state=42)
incoming = clean.drop(reference.index).copy()

# Controlled experiment: simulate a 20% increase
# in incoming monthly charges.
incoming["MonthlyCharges"] *= 1.20

drift_results = {}
for feature in features:
    result = ks_2samp(reference[feature], incoming[feature])
    drift_results[feature] = {
        "ks_statistic": round(float(result.statistic), 5),
        "p_value": float(result.pvalue),
        "drift_detected": bool(result.pvalue < 0.05),
    }

# Calculate SHAP importance on a reproducible sample
model = joblib.load(MODEL_PATH)
X = df.drop(columns=["customerID", "Churn"])
X_sample = X.sample(n=min(200, len(X)), random_state=42)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]
X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_explained = pd.DataFrame(
    X_transformed,
    columns=preprocessor.get_feature_names_out()
)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_explained)

importance = (
    pd.Series(
        abs(shap_values.values).mean(axis=0),
        index=X_explained.columns,
    )
    .sort_values(ascending=False)
    .head(10)
)

evidence = {
    "experiment": (
        "Synthetic drift experiment; MonthlyCharges "
        "in the incoming batch were increased by 20%."
    ),
    "significance_level": 0.05,
    "drift_results": drift_results,
    "top_shap_features": {
        str(name): round(float(value), 5)
        for name, value in importance.items()
    },
    "important_limitations": [
        "This is a static dataset, not live production data.",
        "The drift was deliberately simulated.",
        "SHAP shows model contributions, not causal effects.",
        "Drift alone does not prove performance degradation.",
        "No retraining decision is made by this script.",
    ],
}

prompt = f"""
You are the diagnostic reporting component of DataDrift-Shield.

Explain the following measured evidence for a technical project report.
Use only the supplied facts. Do not invent results or claim causation.

Explain:
1. Which features show statistically significant drift at alpha=0.05.
2. What the KS statistics and p-values mean in plain language.
3. What the top SHAP features tell us about model predictions.
4. What can and cannot be concluded from this experiment.
5. What should be checked before deciding whether retraining is needed.

Distinguish distribution drift from model performance degradation.
Do not claim model performance has declined because no new performance
measurement was supplied. Do not autonomously approve retraining.

Evidence:
{json.dumps(evidence, indent=2)}
"""

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

if not response.text:
    raise RuntimeError("Gemini returned an empty report.")

report_path = REPORT_DIR / "gemini_diagnostic_report.md"
report_path.write_text(response.text, encoding="utf-8")

print("\n--- GEMINI DIAGNOSTIC REPORT ---\n")
print(response.text)
print(f"\nReport saved to: {report_path}")
