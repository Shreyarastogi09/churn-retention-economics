"""
dashboard.py
────────────
CLV-driven Churn-Retention Economics Dashboard (Streamlit)

Presents model outputs as business-facing economics:
  • Executive KPI summary
  • Risk-segment economics with retention-budget allocation
  • Revenue-at-risk waterfall
  • Individual customer deep-dive
  • Feature importance (business drivers)

Run:  streamlit run dashboard.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ARTIFACTS_DIR, RISK_LABELS, RETENTION_COST_MAP, INTERVENTION_LIFT_MAP, CLV_HORIZON_MONTHS

# ─────────────────────────────── Page config ─────────────────────────────────
st.set_page_config(
    page_title="Churn Retention Economics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────── Custom CSS ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* ── KPI Cards ──────────────────────────────────────────────────── */
.kpi-container {
    display: flex;
    gap: 16px;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.kpi-card {
    flex: 1 1 200px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e0e7ff, #c7d2fe, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.73rem;
    color: #64748b;
    margin-top: 4px;
}

/* ── Segment cards ──────────────────────────────────────────────── */
.seg-container { display: flex; gap: 14px; margin: 1rem 0; flex-wrap: wrap; }
.seg-card {
    flex: 1 1 220px;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.06);
}
.seg-critical { background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%); }
.seg-high     { background: linear-gradient(135deg, #431407 0%, #78350f 100%); }
.seg-medium   { background: linear-gradient(135deg, #1a2e05 0%, #365314 100%); }
.seg-low      { background: linear-gradient(135deg, #0c2340 0%, #1e3a5f 100%); }
.seg-title    { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em;
                text-transform: uppercase; margin-bottom: 8px; }
.seg-critical .seg-title { color: #fca5a5; }
.seg-high     .seg-title { color: #fdba74; }
.seg-medium   .seg-title { color: #a3e635; }
.seg-low      .seg-title { color: #7dd3fc; }
.seg-metric   { font-size: 0.75rem; color: #cbd5e1; margin: 4px 0; }
.seg-big      { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }

/* ── Section headers ────────────────────────────────────────────── */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2rem 0 0.8rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.35);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Sidebar styling ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.25rem;
}

/* ── Plotly chart containers ────────────────────────────────────── */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
}

/* ── Tooltip-style info boxes ──────────────────────────────────── */
.info-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 6px;
    border: 1px solid rgba(99,102,241,0.25);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────── Data load ──────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv(ARTIFACTS_DIR / "customer_economics.csv")
    seg = pd.read_csv(ARTIFACTS_DIR / "segment_economics.csv")
    with open(ARTIFACTS_DIR / "model_metrics.json") as f:
        metrics = json.load(f)
    imp = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv",
                      index_col=0, header=None, names=["feature", "importance"])
    return df, seg, metrics, imp


try:
    df, seg, metrics, imp = load_data()
except FileNotFoundError:
    st.error("⚠️  Artifacts not found. Run `python train.py` first.")
    st.stop()


# ──────────────────────────────── Sidebar ────────────────────────────────────
with st.sidebar:
    st.markdown("# 📉 Churn Economics")
    st.markdown("---")
    st.markdown(f"""
    <span class="info-pill">AUC {metrics['roc_auc']}</span>
    <span class="info-pill">F1 {metrics['f1_churn']}</span>
    """, unsafe_allow_html=True)
    st.markdown("")

    view = st.radio(
        "**Navigate**",
        ["🏠 Executive Summary",
         "💰 Retention ROI",
         "🔍 Customer Deep-Dive",
         "⚙️ Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with XGBoost · Scikit-learn · Streamlit")


# ─────────────────────────── Helper functions ────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)

RISK_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f97316",
    "Medium":   "#84cc16",
    "Low":      "#38bdf8",
}

def fmt_dollar(val):
    if abs(val) >= 1_000_000:
        return f"${val/1e6:,.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"

def fmt_pct(val):
    return f"{val:.1%}"


# ═══════════════════════════ 1. EXECUTIVE SUMMARY ════════════════════════════
if view == "🏠 Executive Summary":
    st.markdown("## 📊 Executive Summary")
    st.markdown("*Margin-at-risk and retention opportunity at a glance*")

    # ── KPI row ──────────────────────────────────────────────────
    total_customers   = len(df)
    total_margin_at_risk = df["margin_at_risk"].sum()
    total_clv         = df["clv"].sum()
    avg_churn_prob    = df["churn_prob"].mean()
    critical_count    = (df["risk_segment"] == "Critical").sum() + (df["risk_segment"] == "High").sum()

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-sub">in portfolio</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Margin at Risk</div>
            <div class="kpi-value">{fmt_dollar(total_margin_at_risk)}</div>
            <div class="kpi-sub">{CLV_HORIZON_MONTHS}-month horizon</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total CLV (Margin)</div>
            <div class="kpi-value">{fmt_dollar(total_clv)}</div>
            <div class="kpi-sub">discounted @ 10% APR</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Churn Probability</div>
            <div class="kpi-value">{avg_churn_prob:.1%}</div>
            <div class="kpi-sub">portfolio-wide</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">High + Critical Risk</div>
            <div class="kpi-value">{critical_count:,}</div>
            <div class="kpi-sub">{critical_count/total_customers:.1%} of base</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Segment overview cards ───────────────────────────────────
    st.markdown('<div class="section-header">📋 Risk-Segment Snapshot</div>',
                unsafe_allow_html=True)

    seg_html = '<div class="seg-container">'
    for _, row in seg.iterrows():
        segment = row["risk_segment"]
        css_cls = f"seg-{segment.lower()}"
        seg_html += f"""
        <div class="seg-card {css_cls}">
            <div class="seg-title">{segment} Risk</div>
            <div class="seg-big">{int(row['customer_count']):,}</div>
            <div class="seg-metric">customers</div>
            <div class="seg-metric">Avg churn: {row['avg_churn_prob']:.0%}</div>
            <div class="seg-metric">Margin at risk: {fmt_dollar(row['total_margin_at_risk'])}</div>
            <div class="seg-metric">Median CLV: {fmt_dollar(row['median_clv'])}</div>
        </div>"""
    seg_html += '</div>'
    st.markdown(seg_html, unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Margin at Risk by Segment</div>',
                    unsafe_allow_html=True)
        fig_bar = go.Figure()
        for _, row in seg.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["risk_segment"]],
                y=[row["total_margin_at_risk"]],
                marker_color=RISK_COLORS.get(row["risk_segment"], "#6366f1"),
                text=[fmt_dollar(row["total_margin_at_risk"])],
                textposition="outside",
                name=row["risk_segment"],
                showlegend=False,
            ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(title="Margin at Risk ($)", gridcolor="rgba(99,102,241,0.1)"),
            xaxis=dict(title=""),
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🍩 CLV Distribution by Segment</div>',
                    unsafe_allow_html=True)
        fig_donut = go.Figure(data=[go.Pie(
            labels=seg["risk_segment"],
            values=seg["total_clv"],
            hole=0.55,
            marker=dict(colors=[RISK_COLORS[s] for s in seg["risk_segment"]]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#f1f5f9"),
        )])
        fig_donut.update_layout(**PLOTLY_LAYOUT, height=380,
                                 showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Churn probability distribution ───────────────────────────
    st.markdown('<div class="section-header">📈 Churn Probability Distribution</div>',
                unsafe_allow_html=True)
    fig_hist = px.histogram(
        df, x="churn_prob", nbins=50,
        color_discrete_sequence=["#818cf8"],
        labels={"churn_prob": "Churn Probability"},
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=300,
                            yaxis=dict(title="Customer Count",
                                       gridcolor="rgba(99,102,241,0.1)"))
    # Add vertical lines for segment thresholds
    for threshold in [0.25, 0.50, 0.75]:
        fig_hist.add_vline(x=threshold, line_dash="dash",
                           line_color="rgba(255,255,255,0.3)",
                           annotation_text=f"{threshold:.0%}",
                           annotation_font_color="#94a3b8")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Top churn drivers ────────────────────────────────────────
    st.markdown('<div class="section-header">🔑 Top Churn Drivers</div>',
                unsafe_allow_html=True)
    top_imp = imp.head(12).reset_index()
    top_imp.columns = ["feature", "importance"]
    fig_imp = px.bar(
        top_imp, y="feature", x="importance", orientation="h",
        color="importance",
        color_continuous_scale=["#312e81", "#6366f1", "#a78bfa", "#c084fc"],
    )
    fig_imp.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"),
                           xaxis=dict(title="Importance",
                                      gridcolor="rgba(99,102,241,0.1)"))
    st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════════════ 2. RETENTION ROI ════════════════════════════════
elif view == "💰 Retention ROI":
    st.markdown("## 💰 Retention-Budget Allocation & ROI")
    st.markdown("*Optimal spend allocation across risk segments for maximum return*")

    # ── Budget allocation donut ──────────────────────────────────
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">🎯 Recommended Budget Allocation</div>',
                    unsafe_allow_html=True)

        fig_budget = go.Figure(data=[go.Pie(
            labels=seg["risk_segment"],
            values=seg["recommended_budget_pct"],
            hole=0.6,
            marker=dict(colors=[RISK_COLORS[s] for s in seg["risk_segment"]]),
            textinfo="label+percent",
            textfont=dict(size=14, color="#f1f5f9"),
            hovertemplate="<b>%{label}</b><br>Budget share: %{value:.1f}%<extra></extra>",
        )])
        fig_budget.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False,
                                  annotations=[dict(
                                      text="<b>Budget<br>Split</b>",
                                      x=0.5, y=0.5, font_size=16,
                                      font_color="#a5b4fc",
                                      showarrow=False)])
        st.plotly_chart(fig_budget, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 ROI by Segment</div>',
                    unsafe_allow_html=True)

        fig_roi = go.Figure()
        colors_list = [RISK_COLORS.get(s, "#6366f1") for s in seg["risk_segment"]]
        fig_roi.add_trace(go.Bar(
            x=seg["risk_segment"],
            y=seg["retention_roi"],
            marker_color=colors_list,
            text=[f"{v:.0%}" for v in seg["retention_roi"]],
            textposition="outside",
            textfont=dict(size=14, color="#f1f5f9"),
        ))
        fig_roi.update_layout(**PLOTLY_LAYOUT, height=420,
                               yaxis=dict(title="Retention ROI",
                                          tickformat=".0%",
                                          gridcolor="rgba(99,102,241,0.1)"),
                               xaxis=dict(title=""))
        st.plotly_chart(fig_roi, use_container_width=True)

    # ── Economics breakdown table ────────────────────────────────
    st.markdown('<div class="section-header">📋 Segment Economics Detail</div>',
                unsafe_allow_html=True)

    display_seg = seg[[
        "risk_segment", "customer_count", "avg_churn_prob",
        "total_margin_at_risk", "retention_cost_pp", "total_retention_cost",
        "intervention_lift", "expected_saved_margin", "retention_roi",
        "recommended_budget_pct",
    ]].copy()
    display_seg.columns = [
        "Segment", "Customers", "Avg P(Churn)",
        "Margin at Risk", "Cost/Customer", "Total Cost",
        "Lift", "Expected Saved Margin", "ROI",
        "Budget %",
    ]
    st.dataframe(
        display_seg.style
        .format({
            "Avg P(Churn)": "{:.1%}",
            "Margin at Risk": "${:,.0f}",
            "Cost/Customer": "${:,.0f}",
            "Total Cost": "${:,.0f}",
            "Lift": "{:.0%}",
            "Expected Saved Margin": "${:,.0f}",
            "ROI": "{:.0%}",
            "Budget %": "{:.1f}%",
        }),
        use_container_width=True,
        height=220,
    )

    # ── Waterfall: Saved margin vs cost ───────────────────────────
    st.markdown('<div class="section-header">💧 Margin Recovery Waterfall</div>',
                unsafe_allow_html=True)

    waterfall_items = []
    for _, row in seg.iterrows():
        waterfall_items.append(dict(
            label=f"{row['risk_segment']} Saved",
            value=row["expected_saved_margin"],
        ))
    total_saved = seg["expected_saved_margin"].sum()
    total_cost  = seg["total_retention_cost"].sum()

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(seg) + ["relative", "total"],
        x=[f"{r['risk_segment']}\nSaved" for _, r in seg.iterrows()] + ["Total\nCost", "Net\nBenefit"],
        y=[r["expected_saved_margin"] for _, r in seg.iterrows()] + [-total_cost, 0],
        text=[fmt_dollar(r["expected_saved_margin"]) for _, r in seg.iterrows()] + [
            f"-{fmt_dollar(total_cost)}", fmt_dollar(total_saved - total_cost)
        ],
        textposition="outside",
        connector=dict(line=dict(color="rgba(148,163,184,0.3)")),
        increasing=dict(marker_color="#22c55e"),
        decreasing=dict(marker_color="#ef4444"),
        totals=dict(marker_color="#6366f1"),
        textfont=dict(size=12, color="#f1f5f9"),
    ))
    fig_wf.update_layout(**PLOTLY_LAYOUT, height=420,
                          yaxis=dict(title="Dollars ($)",
                                     gridcolor="rgba(99,102,241,0.1)"))
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Insight box ──────────────────────────────────────────────
    net_benefit = total_saved - total_cost
    st.markdown(f"""
    > **💡 Key Insight:** Investing **{fmt_dollar(total_cost)}** in targeted retention
    > across ROI-positive segments is projected to save **{fmt_dollar(total_saved)}** in contribution margin,
    > yielding a net benefit of **{fmt_dollar(net_benefit)}**
    > ({net_benefit/total_cost:.0%} overall ROI).
    """)


# ═══════════════════════════ 3. CUSTOMER DEEP-DIVE ═══════════════════════════
elif view == "🔍 Customer Deep-Dive":
    st.markdown("## 🔍 Customer Deep-Dive")
    st.markdown("*Explore individual customer economics and risk profiles*")

    # ── Filters ──────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        seg_filter = st.multiselect(
            "Risk Segment", RISK_LABELS,
            default=["Critical", "High"],
        )
    with col_f2:
        min_rev = st.slider(
            "Min Margin at Risk ($)",
            0, int(df["margin_at_risk"].max()),
            value=0,
        )
    with col_f3:
        sort_by = st.selectbox(
            "Sort By",
            ["margin_at_risk", "churn_prob", "clv", "MonthlyCharges"],
            index=0,
        )

    filtered = df[
        df["risk_segment"].isin(seg_filter)
        & (df["margin_at_risk"] >= min_rev)
    ].sort_values(sort_by, ascending=False)

    st.markdown(f"""
    <span class="info-pill">{len(filtered):,} customers</span>
    <span class="info-pill">Margin at Risk: {fmt_dollar(filtered["margin_at_risk"].sum())}</span>
    <span class="info-pill">Total CLV: {fmt_dollar(filtered["clv"].sum())}</span>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Scatter: CLV vs Churn Prob (bubble = MonthlyCharges) ─────
    st.markdown('<div class="section-header">🫧 CLV vs Churn Probability</div>',
                unsafe_allow_html=True)

    scatter_data = filtered.head(2000).copy()  # cap for performance
    fig_scatter = px.scatter(
        scatter_data,
        x="churn_prob", y="clv",
        size="MonthlyCharges",
        color="risk_segment",
        color_discrete_map=RISK_COLORS,
        hover_data=["customerID", "MonthlyCharges", "margin_at_risk", "tenure"],
        labels={
            "churn_prob": "Churn Probability",
            "clv": "Customer Lifetime Value ($)",
            "risk_segment": "Segment",
        },
        size_max=18,
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=480,
                               xaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
                               yaxis=dict(gridcolor="rgba(99,102,241,0.08)"),
                               legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Data table ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Customer Table</div>',
                unsafe_allow_html=True)

    display_cols = [
        "customerID", "risk_segment", "churn_prob", "clv",
        "margin_at_risk", "MonthlyCharges", "TotalCharges",
        "tenure", "Contract", "InternetService",
    ]
    display_df = filtered[display_cols].head(200).copy()
    display_df.columns = [
        "Customer ID", "Segment", "P(Churn)", "CLV ($)",
        "Margin at Risk ($)", "Monthly ($)", "Total ($)",
        "Tenure (mo)", "Contract", "Internet",
    ]
    st.dataframe(
        display_df.style.format({
            "P(Churn)": "{:.1%}",
            "CLV ($)": "${:,.0f}",
            "Margin at Risk ($)": "${:,.0f}",
            "Monthly ($)": "${:,.2f}",
            "Total ($)": "${:,.2f}",
        }).background_gradient(
            subset=["P(Churn)"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True,
        height=500,
    )


# ═══════════════════════════ 4. MODEL PERFORMANCE ════════════════════════════
elif view == "⚙️ Model Performance":
    st.markdown("## ⚙️ Model Performance")
    st.markdown("*Underlying model quality — presented for technical validation*")

    # ── Metric pills ─────────────────────────────────────────────
    pills = "".join([
        f'<span class="info-pill">{k.replace("_", " ").title()}: {v}</span>'
        for k, v in metrics.items()
        if k not in ("confusion_matrix", "test_size", "churn_rate_test")
    ])
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        # ── Confusion matrix heatmap ─────────────────────────────
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>',
                    unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Retained", "Churned"],
            y=["Retained", "Churned"],
            color_continuous_scale=["#1e1b4b", "#4338ca", "#818cf8", "#c7d2fe"],
        )
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        # ── Feature importance (top 15) ──────────────────────────
        st.markdown('<div class="section-header">📊 Feature Importance</div>',
                    unsafe_allow_html=True)
        top_imp = imp.head(15).reset_index()
        top_imp.columns = ["feature", "importance"]
        fig_fimp = px.bar(
            top_imp, y="feature", x="importance", orientation="h",
            color="importance",
            color_continuous_scale=["#312e81", "#6366f1", "#a78bfa"],
        )
        fig_fimp.update_layout(**PLOTLY_LAYOUT, height=380,
                                coloraxis_showscale=False,
                                yaxis=dict(autorange="reversed"),
                                xaxis=dict(title="Importance",
                                           gridcolor="rgba(99,102,241,0.1)"))
        st.plotly_chart(fig_fimp, use_container_width=True)

    # ── Model details ────────────────────────────────────────────
    st.markdown('<div class="section-header">ℹ️ Model Details</div>',
                unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Algorithm | XGBoost Classifier |
        | Test Size | {metrics['test_size']:,} samples |
        | Test Churn Rate | {metrics['churn_rate_test']:.1%} |
        | ROC AUC | **{metrics['roc_auc']}** |
        | F1 (Churn) | **{metrics['f1_churn']}** |
        """)

    with col_d2:
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Precision (Churn) | {metrics['precision_churn']} |
        | Recall (Churn) | {metrics['recall_churn']} |
        | F1 (Retained) | {metrics['f1_retained']} |
        | Accuracy | {metrics['accuracy']} |
        | Avg Precision | {metrics['avg_precision']} |
        """)
