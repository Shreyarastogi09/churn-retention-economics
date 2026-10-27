"""
retention_economics.py
──────────────────────
Translate churn probabilities into dollar-denominated business economics:
  • Customer Lifetime Value (CLV)   – contribution-margin basis
  • Margin-at-Risk per customer & segment
  • Retention ROI modelling per segment
"""
import numpy as np
import pandas as pd

from config import (
    DISCOUNT_RATE, CONTRIBUTION_MARGIN_RATE, CLV_HORIZON_MONTHS,
    RISK_BINS, RISK_LABELS,
    RETENTION_COST_MAP, INTERVENTION_LIFT_MAP,
)


# ── Customer Lifetime Value ─────────────────────────────────────────────────

def compute_clv(monthly_charges: pd.Series,
                churn_prob: np.ndarray,
                margin_rate: float = CONTRIBUTION_MARGIN_RATE,
                discount_rate: float = DISCOUNT_RATE,
                horizon_months: int = CLV_HORIZON_MONTHS) -> pd.Series:
    """
    Discounted CLV on a contribution-margin basis.

    CLV = Σ_{t=1}^{T}  (monthly_margin) × (1 − p_monthly)^t / (1 + r/12)^t

    The model's churn probability is treated as the probability of churning
    **within the full horizon** (a modelling assumption — see config.py).
    We convert it to an implied monthly churn rate:

        p_monthly = 1 − (1 − p_churn)^(1 / horizon_months)

    Parameters
    ----------
    monthly_charges : monthly charge per customer
    churn_prob      : predicted probability of churn over the full horizon
    margin_rate     : contribution-margin fraction (NOT top-line revenue)
    discount_rate   : annual discount rate
    horizon_months  : forecast horizon in months

    Returns
    -------
    pd.Series of CLV in contribution-margin dollars.
    """
    # Convert horizon-level probability to implied monthly rate
    p_monthly = 1 - (1 - churn_prob) ** (1.0 / horizon_months)

    monthly_margin   = monthly_charges * margin_rate
    monthly_discount = discount_rate / 12.0

    clv = pd.Series(0.0, index=monthly_charges.index)
    for t in range(1, horizon_months + 1):
        survival   = (1 - p_monthly) ** t          # prob still a customer
        pv_factor  = 1 / (1 + monthly_discount) ** t
        clv       += monthly_margin * survival * pv_factor

    return clv.round(2)


# ── Margin-at-Risk ───────────────────────────────────────────────────────────

def compute_margin_at_risk(clv: pd.Series,
                           churn_prob: np.ndarray,
                           monthly_charges: pd.Series,
                           margin_rate: float = CONTRIBUTION_MARGIN_RATE,
                           discount_rate: float = DISCOUNT_RATE,
                           horizon_months: int = CLV_HORIZON_MONTHS,
                           ) -> pd.Series:
    """
    Margin-at-Risk = CLV_if_retained − CLV.

    CLV_if_retained is the CLV assuming zero churn (full retention for the
    entire horizon).  Subtracting the churn-adjusted CLV gives the expected
    loss of contribution margin attributable to churn risk — without
    double-counting.

    Parameters
    ----------
    clv             : churn-adjusted CLV (from compute_clv)
    churn_prob      : predicted churn probability (unused directly, but
                      kept in signature for API consistency)
    monthly_charges : monthly charge per customer (needed to compute the
                      zero-churn baseline)
    margin_rate     : contribution-margin fraction
    discount_rate   : annual discount rate
    horizon_months  : forecast horizon in months

    Returns
    -------
    pd.Series of margin-at-risk in contribution-margin dollars.
    """
    clv_retained = compute_clv(
        monthly_charges,
        np.zeros(len(monthly_charges)),
        margin_rate=margin_rate,
        discount_rate=discount_rate,
        horizon_months=horizon_months,
    )
    return (clv_retained - clv).round(2)


# Backward-compatible alias — used during the dashboard transition.
def compute_revenue_at_risk(clv, churn_prob, monthly_charges=None,
                            **kwargs):
    """Deprecated — use compute_margin_at_risk instead."""
    if monthly_charges is None:
        raise TypeError(
            "compute_revenue_at_risk now requires monthly_charges. "
            "Use compute_margin_at_risk directly."
        )
    return compute_margin_at_risk(clv, churn_prob, monthly_charges, **kwargs)


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
        churn_prob, risk_segment, clv, margin_at_risk, MonthlyCharges
    """
    seg = (
        df.groupby("risk_segment", observed=False)
        .agg(
            customer_count    = ("churn_prob", "size"),
            avg_churn_prob    = ("churn_prob", "mean"),
            median_clv        = ("clv", "median"),
            total_clv         = ("clv", "sum"),
            total_margin_at_risk = ("margin_at_risk", "sum"),
            avg_monthly_rev   = ("MonthlyCharges", "mean"),
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

    # Expected saved margin = lift × margin-at-risk
    seg["expected_saved_margin"] = (
        seg["intervention_lift"] * seg["total_margin_at_risk"]
    ).round(2)

    # ROI = (saved_margin − cost) / cost
    seg["retention_roi"] = (
        (seg["expected_saved_margin"] - seg["total_retention_cost"])
        / seg["total_retention_cost"]
    ).round(2)

    # Budget-share recommendation — only allocate to ROI-positive segments
    roi_positive = seg["retention_roi"] > 0
    positive_saved = seg.loc[roi_positive, "expected_saved_margin"].sum()

    seg["recommended_budget_pct"] = 0.0
    if positive_saved > 0:
        seg.loc[roi_positive, "recommended_budget_pct"] = (
            seg.loc[roi_positive, "expected_saved_margin"]
            / positive_saved * 100
        ).round(1)

    # Round dollars
    for col in ["median_clv", "total_clv", "total_margin_at_risk",
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
    df["margin_at_risk"]  = compute_margin_at_risk(
        df["clv"], churn_probs, df["MonthlyCharges"]
    )
    # Backward-compatible alias for dashboard until labels are updated
    df["revenue_at_risk"] = df["margin_at_risk"]
    return df
