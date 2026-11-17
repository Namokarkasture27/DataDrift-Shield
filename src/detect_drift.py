
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"

# Load and prepare the dataset
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

features = ["tenure", "MonthlyCharges", "TotalCharges"]
df = df[features].dropna()

# Create reference and incoming batches
# Fixed random state makes the experiment reproducible.
reference = df.sample(frac=0.70, random_state=42)
incoming = df.drop(reference.index).copy()

# Control experiment: compare unmodified incoming data
# with the reference distribution.
print("\n--- CONTROL: ORIGINAL INCOMING DATA ---")

for feature in features:
    result = ks_2samp(reference[feature], incoming[feature])
    status = "DRIFT" if result.pvalue < 0.05 else "NO DRIFT"

    print(
        f"{feature}: KS={result.statistic:.4f}, "
        f"p-value={result.pvalue:.6f}, {status}"
    )

# Synthetic experiment: deliberately shift monthly charges.
shifted = incoming.copy()
shifted["MonthlyCharges"] *= 1.20

print("\n--- EXPERIMENT: SHIFTED MONTHLY CHARGES ---")

for feature in features:
    result = ks_2samp(reference[feature], shifted[feature])
    status = "DRIFT" if result.pvalue < 0.05 else "NO DRIFT"

    print(
        f"{feature}: KS={result.statistic:.4f}, "
        f"p-value={result.pvalue:.6f}, {status}"
    )

print("\nDrift experiment completed.")
