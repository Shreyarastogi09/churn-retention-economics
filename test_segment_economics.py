"""
test_segment_economics.py
─────────────────────────
Focused test: verifies that build_segment_economics produces numeric
cost and lift columns with no NaN values, even when risk_segment
is a pandas Categorical (the real-world scenario).
"""
import numpy as np
import pandas as pd
import pytest

from retention_economics import (
    build_segment_economics, assign_risk_segments,
    compute_clv, compute_margin_at_risk,
)
from config import RISK_LABELS, RETENTION_COST_MAP, INTERVENTION_LIFT_MAP


def _make_sample_df(n: int = 200) -> pd.DataFrame:
    """Build a small synthetic DataFrame that mirrors enrich_customer_data output."""
    rng = np.random.default_rng(42)
    churn_probs = rng.uniform(0.0, 1.0, size=n)
    monthly     = rng.uniform(20.0, 120.0, size=n)

    charges = pd.Series(monthly)
    clv = compute_clv(charges, churn_probs)
    mar = compute_margin_at_risk(clv, churn_probs, charges)

    df = pd.DataFrame({
        "churn_prob":      churn_probs,
        "risk_segment":    assign_risk_segments(churn_probs),   # ← Categorical!
        "clv":             clv,
        "margin_at_risk":  mar,
        "MonthlyCharges":  monthly,
    })
    return df


class TestBuildSegmentEconomics:
    """Suite for build_segment_economics."""

    def setup_method(self):
        self.df  = _make_sample_df()
        self.seg = build_segment_economics(self.df)

    # ── Core regression: categorical mapping must not crash ──────
    def test_no_crash_on_categorical_input(self):
        """The input risk_segment is Categorical; function must not raise."""
        assert self.df["risk_segment"].dtype.name == "category"
        # If we got here, build_segment_economics didn't crash.

    # ── Mapped columns are numeric, no NaN ──────────────────────
    def test_retention_cost_pp_is_numeric_no_nan(self):
        assert pd.api.types.is_float_dtype(self.seg["retention_cost_pp"])
        assert self.seg["retention_cost_pp"].notna().all()

    def test_intervention_lift_is_numeric_no_nan(self):
        assert pd.api.types.is_float_dtype(self.seg["intervention_lift"])
        assert self.seg["intervention_lift"].notna().all()

    def test_total_retention_cost_is_numeric_no_nan(self):
        assert pd.api.types.is_numeric_dtype(self.seg["total_retention_cost"])
        assert self.seg["total_retention_cost"].notna().all()

    def test_expected_saved_margin_is_numeric_no_nan(self):
        assert pd.api.types.is_numeric_dtype(self.seg["expected_saved_margin"])
        assert self.seg["expected_saved_margin"].notna().all()

    def test_retention_roi_is_numeric_no_nan(self):
        assert pd.api.types.is_numeric_dtype(self.seg["retention_roi"])
        assert self.seg["retention_roi"].notna().all()

    # ── All four risk segments are present ───────────────────────
    def test_all_segments_present(self):
        assert set(self.seg["risk_segment"]) == set(RISK_LABELS)

    # ── Map keys match config labels ─────────────────────────────
    def test_config_keys_match_labels(self):
        assert set(RETENTION_COST_MAP.keys()) == set(RISK_LABELS)
        assert set(INTERVENTION_LIFT_MAP.keys()) == set(RISK_LABELS)

    # ── Budget percentages sum to ~100% (only for ROI+ segments) ─
    def test_budget_pct_sums_to_100(self):
        total = self.seg["recommended_budget_pct"].sum()
        assert abs(total - 100.0) < 0.5 or total == 0.0, (
            f"Budget % sums to {total}, expected ~100 or 0"
        )

    # ── ROI values are plausible (positive for at least one) ─────
    def test_at_least_one_positive_roi(self):
        assert (self.seg["retention_roi"] > 0).any()
