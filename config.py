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
DISCOUNT_RATE      = 0.10        # annual discount rate for CLV present-value
AVG_MARGIN_RATE    = 0.30        # gross-margin % on monthly charges
RETENTION_COST_MAP = {            # $/customer cost of intervention per segment
    "Critical":  50.0,
    "High":      30.0,
    "Medium":    15.0,
    "Low":        5.0,
}
# Estimated lift in retention probability from an intervention
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
