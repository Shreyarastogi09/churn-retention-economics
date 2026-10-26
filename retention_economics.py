"""
retention_economics.py
──────────────────────
Translate churn probabilities into dollar-denominated business economics:
  • Customer Lifetime Value (CLV)
  • Revenue-at-Risk per customer & segment
  • Retention ROI modelling per segment
"""
import numpy as np
import pandas as pd

from config import (
    DISCOUNT_RATE, AVG_MARGIN_RATE,
    RISK_BINS, RISK_LABELS,
    RETENTION_COST_MAP, INTERVENTION_LIFT_MAP,
)


# ── Customer Lifetime Value ─────────────────────────────────────────────────

def compute_clv(monthly_charges: pd.Series,
                churn_prob: np.ndarray,
                margin_rate: float = AVG_MARGIN_RATE,
                discount_rate: float = DISCOUNT_RATE,
                horizon_months: int = 36) -> pd.Series:
    """
    Discounted CLV = Σ_{t=1}^{T}  margin × (1 − p_churn)^t  / (1 + r/12)^t

    Parameters
    ----------
    monthly_charges : monthly revenue per customer
    churn_prob      : predicted monthly churn probability (from model)
    margin_rate     : gross-margin fraction
    discount_rate   : annual discount rate
    horizon_months  : forecast horizon

    Returns
    -------
    pd.Series of CLV in dollars.
    """
    monthly_margin   = monthly_charges * margin_rate
    monthly_discount = discount_rate / 12.0

    clv = pd.Series(0.0, index=monthly_charges.index)
    for t in range(1, horizon_months + 1):
        survival   = (1 - churn_prob) ** t          # prob still a customer
        pv_factor  = 1 / (1 + monthly_discount) ** t
        clv       += monthly_margin * survival * pv_factor

    return clv.round(2)


# ── Revenue-at-Risk ──────────────────────────────────────────────────────────

def compute_revenue_at_risk(clv: pd.Series,
                            churn_prob: np.ndarray) -> pd.Series:
    """Revenue-at-Risk = CLV × P(churn)."""
    return (clv * churn_prob).round(2)


# ── Risk segmentation ───────────────────────────────────────────────────────

def assign_risk_segments(churn_prob: np.ndarray) -> pd.Series:
    """Bin churn probabilities into named risk segments."""
    return pd.cut(
        churn_prob,
        bins=RISK_BINS,
        labels=RISK_LABELS,
        include_lowest=True,
    )


# ── Segment-level economics ─────────────────────────────────────────────────

def build_segment_economics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per risk-segment and compute retention-budget ROI.

    Expected columns in *df*:
        churn_prob, risk_segment, clv, revenue_at_risk, MonthlyCharges
    """
    seg = (
        df.groupby("risk_segment", observed=False)
        .agg(
            customer_count   = ("churn_prob", "size"),
            avg_churn_prob   = ("churn_prob", "mean"),
            median_clv       = ("clv", "median"),
            total_clv        = ("clv", "sum"),
            total_rev_at_risk= ("revenue_at_risk", "sum"),
            avg_monthly_rev  = ("MonthlyCharges", "mean"),
        )
        .reset_index()
    )

    # Attach intervention economics
    # Cast to str — groupby preserves the categorical dtype, and
    # .map() on a categorical returns another categorical which
    # does not support arithmetic operations.
    seg["risk_segment"] = seg["risk_segment"].astype(str)
    seg["retention_cost_pp"]  = seg["risk_segment"].map(RETENTION_COST_MAP)
    seg["intervention_lift"]  = seg["risk_segment"].map(INTERVENTION_LIFT_MAP)
    seg["total_retention_cost"] = seg["customer_count"] * seg["retention_cost_pp"]

    # Expected saved revenue = lift × revenue-at-risk
    seg["expected_saved_rev"] = (
        seg["intervention_lift"] * seg["total_rev_at_risk"]
    ).round(2)

    # ROI = (saved_rev − cost) / cost
    seg["retention_roi"] = (
        (seg["expected_saved_rev"] - seg["total_retention_cost"])
        / seg["total_retention_cost"]
    ).round(2)

    # Budget-share recommendation (proportional to expected saved revenue)
    total_saved = seg["expected_saved_rev"].sum()
    seg["recommended_budget_pct"] = (
        (seg["expected_saved_rev"] / total_saved * 100).round(1)
        if total_saved > 0 else 0.0
    )

    # Round dollars
    for col in ["median_clv", "total_clv", "total_rev_at_risk",
                "avg_monthly_rev", "total_retention_cost"]:
        seg[col] = seg[col].round(2)

    seg["avg_churn_prob"] = seg["avg_churn_prob"].round(3)

    return seg


# ── Per-customer economics DataFrame ────────────────────────────────────────

def enrich_customer_data(df: pd.DataFrame,
                         churn_probs: np.ndarray) -> pd.DataFrame:
    """
    Given the original DataFrame and churn probs, attach all economics columns.
    """
    df = df.copy()
    df["churn_prob"]      = churn_probs
    df["risk_segment"]    = assign_risk_segments(churn_probs)
    df["clv"]             = compute_clv(df["MonthlyCharges"], churn_probs)
    df["revenue_at_risk"] = compute_revenue_at_risk(df["clv"], churn_probs)
    return df
