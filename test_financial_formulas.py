"""
test_financial_formulas.py
──────────────────────────
Formula-correctness tests for the financial calculations.

These tests encode the *intended* (fixed) behaviour:
  • CLV converts the model's churn probability to a monthly rate via
    p_monthly = 1 − (1 − p_churn)^(1/H)
  • Margin-at-risk is CLV_if_retained − CLV (no double-count)
  • Budget allocation excludes segments with negative ROI
  • Edge cases: p_churn = 0 and p_churn = 1
"""
import math

import numpy as np
import pandas as pd
import pytest

from config import (
    CLV_HORIZON_MONTHS, CONTRIBUTION_MARGIN_RATE, DISCOUNT_RATE,
    RISK_LABELS, RETENTION_COST_MAP, INTERVENTION_LIFT_MAP,
)
from retention_economics import (
    compute_clv, compute_margin_at_risk, assign_risk_segments,
    build_segment_economics, enrich_customer_data,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hand_clv(monthly_charge, p_churn,
              margin=CONTRIBUTION_MARGIN_RATE,
              discount=DISCOUNT_RATE,
              horizon=CLV_HORIZON_MONTHS):
    """Reference CLV calculated with explicit Python loops — no pandas."""
    p_monthly = 1 - (1 - p_churn) ** (1.0 / horizon)
    monthly_margin = monthly_charge * margin
    monthly_discount = discount / 12.0
    total = 0.0
    for t in range(1, horizon + 1):
        survival  = (1 - p_monthly) ** t
        pv_factor = 1 / (1 + monthly_discount) ** t
        total    += monthly_margin * survival * pv_factor
    return round(total, 2)


def _hand_clv_retained(monthly_charge,
                       margin=CONTRIBUTION_MARGIN_RATE,
                       discount=DISCOUNT_RATE,
                       horizon=CLV_HORIZON_MONTHS):
    """CLV assuming zero churn (full retention)."""
    return _hand_clv(monthly_charge, p_churn=0.0,
                     margin=margin, discount=discount, horizon=horizon)


# ── CLV unit tests ───────────────────────────────────────────────────────────

class TestComputeCLV:

    def test_known_value(self):
        """CLV for a single customer matches a hand-calculated reference."""
        charge = pd.Series([80.0])
        p = np.array([0.5])
        result = compute_clv(charge, p)[0]
        expected = _hand_clv(80.0, 0.5)
        assert result == pytest.approx(expected, abs=0.01)

    def test_zero_churn_gives_full_stream(self):
        """p_churn = 0 → CLV equals the full discounted margin stream."""
        charge = pd.Series([100.0])
        p = np.array([0.0])
        result = compute_clv(charge, p)[0]
        expected = _hand_clv_retained(100.0)
        assert result == pytest.approx(expected, abs=0.01)

    def test_certain_churn_gives_near_zero(self):
        """p_churn = 1 → monthly rate = 1 → survival = 0 from t=1."""
        charge = pd.Series([100.0])
        p = np.array([1.0])
        result = compute_clv(charge, p)[0]
        assert result == pytest.approx(0.0, abs=0.01)

    def test_higher_churn_means_lower_clv(self):
        """CLV is monotonically decreasing in churn probability."""
        charge = pd.Series([75.0, 75.0, 75.0])
        probs = np.array([0.1, 0.5, 0.9])
        clvs = compute_clv(charge, probs)
        assert clvs.iloc[0] > clvs.iloc[1] > clvs.iloc[2]

    def test_margin_rate_scales_linearly(self):
        """Doubling the margin rate doubles CLV (all else equal)."""
        charge = pd.Series([100.0])
        p = np.array([0.3])
        clv_30 = compute_clv(charge, p, margin_rate=0.30)[0]
        clv_60 = compute_clv(charge, p, margin_rate=0.60)[0]
        assert clv_60 == pytest.approx(2 * clv_30, rel=1e-6)


# ── Margin-at-risk tests ────────────────────────────────────────────────────

class TestMarginAtRisk:

    def test_no_double_count(self):
        """margin_at_risk = CLV_retained − CLV.  Must be < CLV_retained."""
        charge = pd.Series([80.0])
        p = np.array([0.5])
        clv = compute_clv(charge, p)
        mar = compute_margin_at_risk(clv, p, charge)
        clv_ret = _hand_clv_retained(80.0)
        assert mar[0] == pytest.approx(clv_ret - clv[0], abs=0.02)

    def test_zero_churn_zero_risk(self):
        """If p_churn = 0, margin at risk must be 0."""
        charge = pd.Series([100.0])
        p = np.array([0.0])
        clv = compute_clv(charge, p)
        mar = compute_margin_at_risk(clv, p, charge)
        assert mar[0] == pytest.approx(0.0, abs=0.01)

    def test_certain_churn_maximum_risk(self):
        """If p_churn = 1, all margin is at risk."""
        charge = pd.Series([100.0])
        p = np.array([1.0])
        clv = compute_clv(charge, p)
        mar = compute_margin_at_risk(clv, p, charge)
        expected = _hand_clv_retained(100.0)
        assert mar[0] == pytest.approx(expected, abs=0.01)

    def test_at_risk_always_less_than_clv_retained(self):
        """For any 0 < p < 1, margin_at_risk < CLV_retained."""
        charge = pd.Series([60.0, 60.0, 60.0])
        probs = np.array([0.1, 0.5, 0.9])
        clv = compute_clv(charge, probs)
        mar = compute_margin_at_risk(clv, probs, charge)
        clv_ret = _hand_clv_retained(60.0)
        for v in mar:
            assert v < clv_ret


# ── Budget allocation tests ─────────────────────────────────────────────────

class TestBudgetAllocation:

    def _make_seg_df(self):
        """Synthetic customer-level data producing segments with varied ROI."""
        rng = np.random.default_rng(99)
        n = 400
        churn_probs = rng.uniform(0.0, 1.0, size=n)
        monthly = rng.uniform(20.0, 120.0, size=n)
        charges = pd.Series(monthly)
        clv = compute_clv(charges, churn_probs)
        mar = compute_margin_at_risk(clv, churn_probs, charges)
        df = pd.DataFrame({
            "churn_prob":      churn_probs,
            "risk_segment":    assign_risk_segments(churn_probs),
            "clv":             clv,
            "margin_at_risk":  mar,
            "MonthlyCharges":  monthly,
        })
        return df

    def test_negative_roi_gets_zero_budget(self):
        """Segments with ROI ≤ 0 must receive 0% budget share."""
        df = self._make_seg_df()
        seg = build_segment_economics(df)
        neg_roi = seg[seg["retention_roi"] <= 0]
        if len(neg_roi) > 0:
            assert (neg_roi["recommended_budget_pct"] == 0.0).all()

    def test_positive_roi_budget_sums_to_100(self):
        """Budget allocated only to ROI-positive segments still sums to ~100."""
        df = self._make_seg_df()
        seg = build_segment_economics(df)
        total = seg["recommended_budget_pct"].sum()
        assert abs(total - 100.0) < 0.5 or total == 0.0


# ── Edge-case regression tests ──────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_monthly_charge(self):
        """Customers with $0 monthly charge should have CLV = 0."""
        charge = pd.Series([0.0])
        p = np.array([0.3])
        clv = compute_clv(charge, p)
        assert clv[0] == pytest.approx(0.0, abs=0.01)

    def test_vectorised_matches_scalar(self):
        """Vectorised result matches individual scalar computations."""
        charges = pd.Series([30.0, 60.0, 90.0])
        probs = np.array([0.2, 0.5, 0.8])
        vec_clvs = compute_clv(charges, probs)
        for i in range(3):
            expected = _hand_clv(charges.iloc[i], probs[i])
            assert vec_clvs.iloc[i] == pytest.approx(expected, abs=0.01)
