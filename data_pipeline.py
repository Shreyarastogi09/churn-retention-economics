"""
data_pipeline.py
────────────────
Load, clean, and feature-engineer the IBM Telco Customer Churn dataset.
Returns train/test splits plus a fully-encoded DataFrame for scoring.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    RAW_DATA_FILE, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN, ID_COLUMN,
)


def load_raw_data(path=None):
    """Read the CSV and do minimal cleaning."""
    path = path or RAW_DATA_FILE
    df = pd.read_csv(path)

    # TotalCharges has blanks for brand-new customers → coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Binary target
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add business-meaningful derived features before encoding.
    """
    df = df.copy()

    # ── Tenure bands ────────────────────────────────────────────────
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, np.inf],
        labels=["0-12m", "12-24m", "24-48m", "48-72m", "72m+"],
    )

    # ── Average monthly revenue (guards div/0) ─────────────────────
    df["avg_monthly_revenue"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # ── Bundled-services count ────────────────────────────────────
    service_cols = [
        "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # ── Has premium support ──────────────────────────────────────
    df["has_premium_support"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    # ── Charge-to-tenure ratio (engagement proxy) ─────────────────
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df


def encode_dataframe(df: pd.DataFrame):
    """
    One-hot encode categoricals, keep numerics, return X, y, feature_names.
    """
    # Combine engineered categoricals
    cat_cols = CATEGORICAL_FEATURES + ["tenure_band"]
    num_cols = NUMERIC_FEATURES + [
        "avg_monthly_revenue", "num_services",
        "has_premium_support", "charge_tenure_ratio",
    ]

    X_num = df[num_cols].copy()
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True).astype(int)

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COLUMN]
    return X, y, list(X.columns)


def prepare_data(path=None, test_size=0.20, random_state=42):
    """
    End-to-end pipeline: load → engineer → encode → split.
    Returns (X_train, X_test, y_train, y_test, df_full, feature_names)
    """
    df = load_raw_data(path)
    df = engineer_features(df)
    X, y, feature_names = encode_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, df, feature_names
