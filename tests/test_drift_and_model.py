"""
Basic tests for the drift-detection logic and the trained baseline model.

Run with: pytest tests/
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "baseline_xgboost.joblib"


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip("Telco-Customer-Churn.csv not present (see README Setup).")
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def test_dataset_has_expected_columns(dataset):
    expected = {"customerID", "Churn", "MonthlyCharges", "TotalCharges", "tenure"}
    assert expected.issubset(set(dataset.columns))


def test_ks_test_flags_no_drift_on_identical_distribution(dataset):
    clean = dataset[["MonthlyCharges"]].dropna()
    sample_a = clean.sample(frac=0.5, random_state=1)
    sample_b = clean.sample(frac=0.5, random_state=1)  # same sample, same seed
    result = ks_2samp(sample_a["MonthlyCharges"], sample_b["MonthlyCharges"])
    assert result.pvalue > 0.05


def test_ks_test_flags_drift_after_synthetic_shift(dataset):
    clean = dataset[["MonthlyCharges"]].dropna()
    reference = clean.sample(frac=0.70, random_state=42)
    incoming = clean.drop(reference.index).copy()
    incoming["MonthlyCharges"] *= 1.20  # same 20% shift used in detect_drift.py

    result = ks_2samp(reference["MonthlyCharges"], incoming["MonthlyCharges"])
    assert result.pvalue < 0.05


def test_baseline_model_predicts_valid_probabilities(dataset):
    if not MODEL_PATH.exists():
        pytest.skip("Baseline model not trained yet (run src/train_baseline.py).")
    model = joblib.load(MODEL_PATH)
    X = dataset.drop(columns=["customerID", "Churn"]).sample(5, random_state=0)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
