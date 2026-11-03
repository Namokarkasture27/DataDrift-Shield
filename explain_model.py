
from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_xgboost.joblib"
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load the trained model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["customerID", "Churn"])

# 2. Select a reproducible sample to explain
X_sample = X.sample(
    n=min(300, len(X)),
    random_state=42
)

# 3. Apply the preprocessing learned by our baseline model
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

# SHAP and plotting work more conveniently with a DataFrame.
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_explained = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

# 4. Calculate SHAP values
print("Calculating SHAP explanations...")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(X_explained)

# 5. Save a global feature-importance plot
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 6. Save a detailed SHAP summary plot
shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 7. Print the most influential features
importance = pd.Series(
    abs(shap_values.values).mean(axis=0),
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 features by mean absolute SHAP value:")
print(importance.head(10).to_string())

print("\nSHAP analysis completed.")
print("Plots saved in:", OUTPUT_DIR)
