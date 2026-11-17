"""
Module 5 — Automated Retraining & Validation (champion/challenger).

1. Loads the current production ("champion") model.
2. Simulates a batch of newer data arriving (here: a resample of the same
   dataset with a mild synthetic shift, standing in for "new" incoming data).
3. Retrains a fresh ("challenger") model on champion-train + new data.
4. Validates both models on the SAME held-out test set, using ROC-AUC.
5. Promotes the challenger to production only if it beats the champion.
   The champion is never replaced automatically unless this check passes.
6. Appends a human-readable line to outputs/retraining_log.csv so there is
   a visible model version history for the dashboard (Module 6) to read.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CHAMPION_PATH = MODEL_DIR / "baseline_xgboost.joblib"
CHALLENGER_PATH = MODEL_DIR / "challenger_xgboost.joblib"
LOG_PATH = OUTPUT_DIR / "retraining_log.csv"


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return X, y


def build_pipeline(X):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    if not CHAMPION_PATH.exists():
        raise RuntimeError(
            "No champion model found. Run src/train_baseline.py first."
        )

    champion = joblib.load(CHAMPION_PATH)
    X, y = load_and_clean()

    # Fixed test split so champion and challenger are judged on identical data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Stand-in for "newer production data": resample the training pool with
    # a different seed, simulating the next batch of customers.
    new_batch = X_train.sample(frac=0.5, random_state=7)
    new_batch_y = y_train.loc[new_batch.index]

    challenger_train_X = pd.concat([X_train, new_batch])
    challenger_train_y = pd.concat([y_train, new_batch_y])

    challenger = build_pipeline(challenger_train_X)
    challenger.fit(challenger_train_X, challenger_train_y)

    champion_pred = champion.predict(X_test)
    champion_proba = champion.predict_proba(X_test)[:, 1]
    challenger_pred = challenger.predict(X_test)
    challenger_proba = challenger.predict_proba(X_test)[:, 1]

    champion_auc = roc_auc_score(y_test, champion_proba)
    challenger_auc = roc_auc_score(y_test, challenger_proba)
    champion_acc = accuracy_score(y_test, champion_pred)
    challenger_acc = accuracy_score(y_test, challenger_pred)

    promoted = challenger_auc > champion_auc

    print(f"Champion  -> accuracy: {champion_acc:.4f}, ROC-AUC: {champion_auc:.4f}")
    print(f"Challenger-> accuracy: {challenger_acc:.4f}, ROC-AUC: {challenger_auc:.4f}")

    if promoted:
        joblib.dump(challenger, CHAMPION_PATH)
        print("\nChallenger beat the champion on ROC-AUC. Promoted to production.")
    else:
        print("\nChallenger did not beat the champion. Champion kept in production.")

    joblib.dump(challenger, CHALLENGER_PATH)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_accuracy": round(float(champion_acc), 4),
        "champion_roc_auc": round(float(champion_auc), 4),
        "challenger_accuracy": round(float(challenger_acc), 4),
        "challenger_roc_auc": round(float(challenger_auc), 4),
        "promoted": promoted,
    }
    log_df = pd.DataFrame([log_entry])
    header = not LOG_PATH.exists()
    log_df.to_csv(LOG_PATH, mode="a", header=header, index=False)
    print(f"\nLogged this run to: {LOG_PATH}")


if __name__ == "__main__":
    main()
