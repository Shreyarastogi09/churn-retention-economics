"""
churn_model.py
──────────────
Train an XGBoost churn classifier, evaluate it, persist the model,
and expose scoring + SHAP explanation helpers.
"""
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score, confusion_matrix
)

from config import XGBOOST_PARAMS, ARTIFACTS_DIR


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val=None, y_val=None):
    """
    Fit an XGBoost classifier.  Returns the fitted model.
    If validation data is supplied, early-stopping is used.
    """
    # Handle class imbalance via scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = xgb.XGBClassifier(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Return a dict of business-relevant metrics and the confusion matrix.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc":            round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision":      round(average_precision_score(y_test, y_proba), 4),
        "f1_churn":           round(report["1"]["f1-score"], 4),
        "precision_churn":    round(report["1"]["precision"], 4),
        "recall_churn":       round(report["1"]["recall"], 4),
        "f1_retained":        round(report["0"]["f1-score"], 4),
        "accuracy":           round(report["accuracy"], 4),
        "confusion_matrix":   cm.tolist(),
        "test_size":          int(len(y_test)),
        "churn_rate_test":    round(y_test.mean(), 4),
    }
    return metrics


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_customers(model, X: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for every row in X."""
    return model.predict_proba(X)[:, 1]


# ── Persistence ──────────────────────────────────────────────────────────────

def save_model(model, metrics: dict, feature_names: list):
    """Persist model + metadata to ARTIFACTS_DIR."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "xgb_churn_model.joblib")
    with open(ARTIFACTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)


def load_model():
    """Load persisted model."""
    return joblib.load(ARTIFACTS_DIR / "xgb_churn_model.joblib")
