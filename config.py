"""
Centralised configuration for the Churn-Retention-Economics project.
All magic numbers, paths, and hyper-parameters live here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
DATA_DIR       = PROJECT_ROOT / "data"
ARTIFACTS_DIR  = PROJECT_ROOT / "artifacts"
RAW_DATA_FILE  = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"   # IBM Telco

# ── Feature engineering ─────────────────────────────────────────────────────
# Columns that stay as-is (numeric)
NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
]

# Columns to one-hot-encode
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

TARGET_COLUMN = "Churn"
ID_COLUMN     = "customerID"

# ── CLV economics ───────────────────────────────────────────────────────────
# Annual discount rate applied to future cash-flows in the CLV calculation.
DISCOUNT_RATE      = 0.10

# ── Churn-label horizon ─────────────────────────────────────────────────────
# MODELLING ASSUMPTION — not a known property of the dataset.
# The IBM Telco "Churn" column is an observed snapshot (did this customer
# churn?) with no explicit calendar period.  We *propose* treating the
# model's predicted churn probability as a probability of churning within
# CLV_HORIZON_MONTHS months and convert it to an implied monthly rate
# inside compute_clv().  Change this value (and re-run train.py) to test
# sensitivity.
CLV_HORIZON_MONTHS = 36

# Assumed contribution-margin fraction applied to MonthlyCharges.
# This is NOT top-line revenue — downstream dollar values represent
# contribution margin, and labels should reflect that.
CONTRIBUTION_MARGIN_RATE = 0.30

# Backward-compatible alias — will be removed once all call-sites are
# updated in subsequent steps.
AVG_MARGIN_RATE = CONTRIBUTION_MARGIN_RATE

# Assumed per-customer cost of a retention intervention, segmented by
# risk tier.  Source: placeholder estimates — replace with actuals when
# available.
RETENTION_COST_MAP = {
    "Critical":  50.0,
    "High":      30.0,
    "Medium":    15.0,
    "Low":        5.0,
}

# Fraction of at-risk margin that can be recovered by a successful
# intervention, per segment.  Despite the name "lift", this is applied
# as a multiplier on dollar margin-at-risk, NOT as a probability delta.
# Source: placeholder estimates — replace with actuals when available.
INTERVENTION_LIFT_MAP = {
    "Critical":  0.15,
    "High":      0.12,
    "Medium":    0.08,
    "Low":       0.03,
}


# ── Model hyper-parameters (XGBoost) ────────────────────────────────────────
XGBOOST_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda         = 1.0,
    eval_metric       = "logloss",
    random_state      = 42,
)

# ── Risk segments (churn-probability bins) ──────────────────────────────────
RISK_BINS   = [0.0, 0.25, 0.50, 0.75, 1.01]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]
