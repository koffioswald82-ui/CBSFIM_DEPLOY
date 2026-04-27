"""
CBSFIM - Customer Behavior & Strategic Financial Impact Modeling
================================================================
Fintech-grade executive dashboard - v2.0 (BI Layer integrated)

Run:  streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config.settings import PROCESSED_DIR, DATA_DIR

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CBSFIM | Financial AI Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    "bg":      "#FFFFFF",
    "surface": "#F6F8FA",
    "border":  "#D0D7DE",
    "primary": "#1A7F37",
    "accent":  "#0969DA",
    "danger":  "#CF222E",
    "warning": "#9A6700",
    "success": "#1A7F37",
    "text":    "#1F2328",
    "subtext": "#57606A",
    "high":    "#CF222E",
    "medium":  "#9A6700",
    "low":     "#1A7F37",
}

SEG_COLORS = {
    "retail":         "#0969DA",
    "premium":        "#8250DF",
    "high_net_worth": "#B08800",
}
SEG_LABEL = {
    "retail":         "Retail Banking",
    "premium":        "Premium Banking",
    "high_net_worth": "High Net Worth",
}
ACTION_COLORS = {
    "Retain - Premium Offer":  "#CF222E",
    "Retain - Standard Offer": "#9A6700",
    "Cross-Sell Opportunity":  "#0969DA",
    "Monitor":                 "#57606A",
    "No Action":               "#1A7F37",
}
URGENCY_COLORS = {
    "CRITICAL": "#CF222E",
    "HIGH":     "#9A6700",
    "MEDIUM":   "#1F6FEB",
    "LOW":      "#8B949E",
    "MINIMAL":  "#1A7F37",
}
RISK_COLORS = {"High": C["high"], "Medium": C["warning"], "Low": C["low"]}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html, body, [data-testid="stApp"] {{
    background-color: {C["bg"]}; color: {C["text"]};
}}
section[data-testid="stSidebar"] {{
    background-color: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}
[data-testid="metric-container"] {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 10px;
    padding: 18px 20px;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: {C["text"]} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {C["subtext"]} !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
h1, h2, h3 {{ color: {C["text"]} !important; }}
.stDataFrame {{ background: {C["surface"]}; }}
div[data-baseweb="tab-list"] {{ background: {C["surface"]}; border-radius: 8px; }}
div[data-baseweb="tab"] {{ color: {C["subtext"]}; }}
div[data-baseweb="tab"][aria-selected="true"] {{
    color: {C["text"]} !important;
    border-bottom: 2px solid {C["accent"]} !important;
}}
.badge-critical {{ background:#FFEBE9; color:{C["high"]};    border:1px solid {C["high"]};    border-radius:5px; padding:2px 9px; font-size:0.78rem; font-weight:700; }}
.badge-high     {{ background:#FFF8C5; color:{C["warning"]}; border:1px solid {C["warning"]}; border-radius:5px; padding:2px 9px; font-size:0.78rem; font-weight:600; }}
.badge-medium   {{ background:#DDF4FF; color:{C["accent"]};  border:1px solid {C["accent"]};  border-radius:5px; padding:2px 9px; font-size:0.78rem; font-weight:600; }}
.badge-low      {{ background:#f0f0f0; color:{C["subtext"]}; border:1px solid {C["border"]};  border-radius:5px; padding:2px 9px; font-size:0.78rem; font-weight:500; }}
.badge-risk-high   {{ background:#FFEBE9; color:{C["high"]};    border:1px solid {C["high"]};    border-radius:5px; padding:2px 8px; font-size:0.78rem; font-weight:600; }}
.badge-risk-medium {{ background:#FFF8C5; color:{C["warning"]}; border:1px solid {C["warning"]}; border-radius:5px; padding:2px 8px; font-size:0.78rem; font-weight:600; }}
.badge-risk-low    {{ background:#DAFBE1; color:{C["low"]};     border:1px solid {C["low"]};     border-radius:5px; padding:2px 8px; font-size:0.78rem; font-weight:600; }}
hr {{ border-color: {C["border"]}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor ="#F6F8FA",
    font=dict(color=C["text"], family="Inter, Arial, sans-serif", size=12),
    margin=dict(t=40, b=30, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
)

BI_DIR      = DATA_DIR / "bi"
TABLEAU_DIR = BI_DIR / "tableau"
KPI_DIR     = BI_DIR / "kpis"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def load_customers() -> pd.DataFrame | None:
    """Loads enriched customer table (BI layer) or falls back to recommendations."""
    bi_path  = TABLEAU_DIR / "customer_table.parquet"
    raw_path = PROCESSED_DIR / "recommendations.parquet"
    if bi_path.exists():
        return normalize_columns(pd.read_parquet(bi_path))
    if raw_path.exists():
        return pd.read_parquet(raw_path)
    return None

@st.cache_data(ttl=600)
def load_time_series() -> pd.DataFrame | None:
    p = TABLEAU_DIR / "time_series.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=600)
def load_financial_aggregates() -> pd.DataFrame | None:
    p = TABLEAU_DIR / "financial_aggregates.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=600)
def load_kpi_report() -> dict:
    p = KPI_DIR / "kpi_report.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@st.cache_data(ttl=600)
def load_metrics() -> dict:
    p = PROCESSED_DIR / "business_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _fmt(v, prefix="", suffix="", decimals=0):
    if v is None:
        return "-"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def _has_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def _f(v) -> float:
    """Cast any value (numpy scalar, string, int) to Python float safely."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names so page functions work whether data comes from
    customer_table.parquet (BI layer) or recommendations.parquet (raw ML).
    """
    renames = {
        "total_revenue_generated": "predicted_revenue_annual",
        "clv_5yr":                 "clv",
    }
    return df.rename(columns={k: v for k, v in renames.items() if k in df.columns})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    bi_available = _has_col(df, "engagement_index")

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:16px 0 8px 0;">
            <div style="font-size:1.3rem; font-weight:700; color:{C['text']};">🏦 CBSFIM</div>
            <div style="font-size:0.75rem; color:{C['subtext']}; margin-top:2px;">
                Financial AI Platform &nbsp;·&nbsp; v2.0
                {"&nbsp;·&nbsp; <span style='color:#3FB950;'>BI ✓</span>" if bi_available else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown(f"<span style='color:{C['subtext']}; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em;'>Filters</span>", unsafe_allow_html=True)

        segments = sorted(df["segment"].unique().tolist())
        sel_seg  = st.multiselect("Segment", segments, default=segments,
                                  format_func=lambda x: SEG_LABEL.get(x, x))

        risks    = ["High", "Medium", "Low"]
        sel_risk = st.multiselect("Risk Level", risks, default=risks)

        if _has_col(df, "region"):
            regions  = sorted(df["region"].unique().tolist())
            sel_reg  = st.multiselect("Region", regions, default=regions)
        else:
            sel_reg = None

        inc_min = float(df["income"].min())
        inc_max = float(df["income"].max())
        sel_inc = st.slider("Income Range ($)", int(inc_min), int(inc_max),
                            (int(inc_min), int(inc_max)), step=5_000)

        actions  = sorted(df["recommended_action"].unique().tolist())
        sel_act  = st.multiselect("Recommended Action", actions, default=actions)

        st.divider()
        if bi_available:
            st.markdown(f"<span style='color:{C['subtext']}; font-size:0.73rem;'>BI Layer active - {len(df):,} customers enriched</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:{C['warning']}; font-size:0.73rem;'>⚠ Run run_bi_pipeline.py to unlock all features</span>", unsafe_allow_html=True)

    mask = (
        df["segment"].isin(sel_seg)
        & df["risk_level"].isin(sel_risk)
        & df["income"].between(*sel_inc)
        & df["recommended_action"].isin(sel_act)
    )
    if sel_reg is not None and _has_col(df, "region"):
        mask &= df["region"].isin(sel_reg)

    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - Executive Summary
# ══════════════════════════════════════════════════════════════════════════════

def page_executive_summary(df: pd.DataFrame, kpi: dict):
    st.markdown("<h2 style='margin-top:0;'>Executive Summary</h2>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{C['subtext']};'>Strategic overview - {len(df):,} customers in view</span>", unsafe_allow_html=True)
    st.markdown("")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    summ = kpi.get("summary", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    total_rev = _f(df["predicted_revenue_annual"].sum()) if _has_col(df, "predicted_revenue_annual") \
        else _f(summ.get("total_revenue", 0))

    with c1:
        st.metric("Total Revenue", f"${total_rev:,.0f}")
    with c2:
        if _has_col(df, "revenue_at_risk"):
            rar = _f(df["revenue_at_risk"].sum())
        elif _has_col(df, "predicted_revenue_annual"):
            rar = _f(df.loc[df["risk_level"] == "High", "predicted_revenue_annual"].sum())
        else:
            rar = _f(summ.get("total_revenue_at_risk", 0))
        rar_pct = rar / total_rev * 100 if total_rev else 0
        st.metric("Revenue at Risk", f"${rar:,.0f}", f"{rar_pct:.1f}% of revenue")
    with c3:
        churn_pct = (df["risk_level"] == "High").mean() * 100
        st.metric("Churn Rate", f"{churn_pct:.1f}%")
    with c4:
        total_clv = _f(df["clv"].sum()) if _has_col(df, "clv") else 0.0
        st.metric("5-yr CLV", f"${total_clv:,.0f}")
    with c5:
        hvar = len(df[(df["risk_level"] == "High") & (df["financial_value"] == "High")])
        st.metric("Critical Accounts", f"{hvar:,}", "⚠ High-value + High-risk")
    with c6:
        if _has_col(df, "net_profit_per_client"):
            net = _f(df["net_profit_per_client"].sum())
            st.metric("Net Portfolio Profit", f"${net:,.0f}")
        elif _has_col(df, "predicted_revenue_annual"):
            exp_loss = _f((df["predicted_revenue_annual"] * df["churn_probability"]).sum())
            st.metric("Expected Loss", f"${exp_loss:,.0f}")

    st.markdown("")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Risk Level Distribution")
        rc = df["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        fig = px.pie(rc, names="Risk Level", values="Count",
                     color="Risk Level", color_discrete_map=RISK_COLORS, hole=0.55)
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Revenue at Risk by Segment")
        rar_col_seg = "revenue_at_risk" if _has_col(df, "revenue_at_risk") else "predicted_revenue_annual"
        seg_rar = (
            df[df["risk_level"] == "High"]
            .groupby("segment")[rar_col_seg].sum().reset_index()
            .rename(columns={rar_col_seg: "Revenue at Risk"})
            .sort_values("Revenue at Risk", ascending=True)
        )
        fig2 = px.bar(seg_rar, x="Revenue at Risk", y="segment", orientation="h",
                      color="segment", color_discrete_map=SEG_COLORS, labels={"segment": ""})
        fig2.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Churn distribution violin
    st.markdown("#### Churn Probability Distribution by Segment")
    fig3 = px.violin(df, y="churn_probability", x="segment",
                     color="segment", color_discrete_map=SEG_COLORS,
                     box=True, points=False,
                     labels={"churn_probability": "Churn Probability", "segment": ""})
    fig3.update_layout(**PLOT_LAYOUT, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Engagement index (BI layer only)
    if _has_col(df, "engagement_index"):
        st.markdown("#### Engagement Index by Segment × Risk Level")
        fig4 = px.box(df, x="segment", y="engagement_index", color="risk_level",
                      color_discrete_map=RISK_COLORS,
                      labels={"engagement_index": "Engagement Index [0–1]", "segment": ""},
                      category_orders={"risk_level": ["High", "Medium", "Low"]})
        fig4.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - Customer Risk Explorer
# ══════════════════════════════════════════════════════════════════════════════

def page_customer_risk_explorer(df: pd.DataFrame):
    st.markdown("<h2 style='margin-top:0;'>Customer Risk Explorer</h2>", unsafe_allow_html=True)

    search = st.text_input("Search by Customer ID", placeholder="e.g. CID_00001234")

    if search:
        row = df[df["customer_id"].str.contains(search.strip(), case=False)]
        if row.empty:
            st.warning("No customer found.")
        else:
            r = row.iloc[0]
            risk_cls = f"badge-risk-{r['risk_level'].lower()}"
            st.markdown("---")
            st.markdown(f"### {r['customer_id']}")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Profile**")
                st.markdown(f"Segment: `{SEG_LABEL.get(r['segment'], r['segment'])}`")
                st.markdown(f"Age: {r['age']}")
                st.markdown(f"Income: ${r['income']:,.0f}")
                st.markdown(f"Balance: ${r['account_balance']:,.0f}")
                st.markdown(f"Tenure: {r.get('tenure_months', r.get('customer_tenure_years', '-'))} {'months' if 'tenure_months' in r.index else 'years'}")
                if _has_col(df, "region"):
                    st.markdown(f"Region: `{r.get('region', '-')}`")
            with c2:
                st.metric("Churn Probability", f"{r['churn_probability']:.1%}")
                st.metric("Annual Revenue", f"${r['predicted_revenue_annual']:,.0f}")
                st.metric("5-yr CLV", f"${r['clv']:,.0f}")
            with c3:
                if _has_col(df, "net_profit_per_client"):
                    st.metric("Net Profit/yr", f"${r['net_profit_per_client']:,.0f}")
                if _has_col(df, "revenue_at_risk"):
                    st.metric("Revenue at Risk", f"${r['revenue_at_risk']:,.0f}")
                if _has_col(df, "engagement_index"):
                    st.metric("Engagement Index", f"{r['engagement_index']:.3f}")
            with c4:
                st.markdown(f"**Risk:** <span class='{risk_cls}'>{r['risk_level']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Value:** `{r['financial_value']}`")
                st.markdown(f"**Action:** `{r['recommended_action']}`")
                st.markdown(f"**Priority:** `{r['priority_score']:.4f}`")
                if _has_col(df, "value_tier"):
                    st.markdown(f"**Value Tier:** `{r.get('value_tier', '-')}`")
                if _has_col(df, "inactivity_streak"):
                    st.markdown(f"**Inactivity Streak:** `{int(r.get('inactivity_streak', 0))} months`")
                if _has_col(df, "days_since_last_transaction"):
                    st.markdown(f"**Last Transaction:** `{int(r.get('days_since_last_transaction', 0))} days ago`")

            if "action_rationale" in r.index:
                st.info(f"**AI Rationale:** {r['action_rationale']}")

    st.markdown("---")
    st.markdown("#### Browse Customers")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sort_options = ["priority_score", "churn_probability", "clv",
                        "predicted_revenue_annual"]
        if _has_col(df, "revenue_at_risk"):
            sort_options.insert(2, "revenue_at_risk")
        sort_col = st.selectbox("Sort by", sort_options)
    with col2:
        top_n = st.selectbox("Show top N", [50, 100, 250, 500])
    with col3:
        risk_filter = st.selectbox("Risk filter", ["All", "High", "Medium", "Low"])
    with col4:
        if _has_col(df, "action_urgency"):
            urg_opts = ["All"] + sorted(df["action_urgency"].unique().tolist())
            urg_filter = st.selectbox("Urgency", urg_opts)
        else:
            urg_filter = "All"

    view = df.copy()
    if risk_filter != "All":
        view = view[view["risk_level"] == risk_filter]
    if urg_filter != "All" and _has_col(view, "action_urgency"):
        view = view[view["action_urgency"] == urg_filter]
    view = view.nlargest(top_n, sort_col)

    base_cols   = ["customer_id", "segment", "age", "income",
                   "churn_probability", "predicted_revenue_annual", "clv",
                   "risk_level", "financial_value", "recommended_action", "priority_score"]
    extra_cols  = [c for c in ["revenue_at_risk", "engagement_index",
                               "net_profit_per_client", "value_tier"]
                   if _has_col(view, c)]
    display_cols = base_cols + extra_cols
    display_cols = [c for c in display_cols if c in view.columns]

    fmt_dict = {
        "income": "{:,.0f}", "churn_probability": "{:.2%}",
        "predicted_revenue_annual": "{:,.0f}", "clv": "{:,.0f}",
        "priority_score": "{:.4f}",
    }
    if "revenue_at_risk" in display_cols:
        fmt_dict["revenue_at_risk"] = "{:,.0f}"
    if "engagement_index" in display_cols:
        fmt_dict["engagement_index"] = "{:.3f}"
    if "net_profit_per_client" in display_cols:
        fmt_dict["net_profit_per_client"] = "{:,.0f}"

    styled = view[display_cols].style.format(fmt_dict)
    if "churn_probability" in display_cols:
        styled = styled.background_gradient(subset=["churn_probability"], cmap="Reds")
    if "clv" in display_cols:
        styled = styled.background_gradient(subset=["clv"], cmap="Greens")

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Scatter: Value vs Risk
    st.markdown("#### Value vs Risk - Scatter Plot")
    sample = df.sample(min(8000, len(df)), random_state=42)
    x_col  = "churn_probability_pct" if _has_col(df, "churn_probability_pct") else "churn_probability"
    y_col  = "clv"
    size_c = "revenue_at_risk"        if _has_col(df, "revenue_at_risk")       else None

    fig = px.scatter(
        sample, x=x_col, y=y_col,
        color="risk_level", color_discrete_map=RISK_COLORS,
        size=size_c, size_max=12, opacity=0.5,
        symbol="segment",
        labels={x_col: "Churn Probability (%)", y_col: "5-yr CLV ($)", "risk_level": "Risk"},
        hover_data=["customer_id", "recommended_action", "priority_score"],
    )
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - Financial Impact
# ══════════════════════════════════════════════════════════════════════════════

def page_financial_impact(df: pd.DataFrame, fa: pd.DataFrame | None):
    st.markdown("<h2 style='margin-top:0;'>Financial Impact Analysis</h2>", unsafe_allow_html=True)

    # ── Profitability heatmap (BI layer) ──────────────────────────────────────
    if fa is not None:
        seg_reg = fa[fa["aggregation_level"] == "by_segment_region"].copy()
        if not seg_reg.empty and "segment" in seg_reg.columns and "region" in seg_reg.columns:
            st.markdown("#### Profitability Heatmap - Segment × Region")
            pivot = (
                seg_reg.pivot(index="segment", columns="region",
                              values="profitability_ratio")
                .fillna(0)
            )
            fig_heat = px.imshow(
                pivot.round(1),
                color_continuous_scale=[[0, C["danger"]], [0.5, C["warning"]], [1, C["success"]]],
                text_auto=".1f",
                labels=dict(x="Region", y="Segment", color="Profit %"),
                aspect="auto",
            )
            fig_heat.update_layout(**PLOT_LAYOUT, coloraxis_colorbar_title="Profit %")
            st.plotly_chart(fig_heat, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Expected Revenue Loss Distribution")
        df_plot = df.copy()
        if not _has_col(df_plot, "revenue_at_risk"):
            df_plot["revenue_at_risk"] = df_plot["predicted_revenue_annual"] * df_plot["churn_probability"]
        fig = px.histogram(df_plot, x="revenue_at_risk", nbins=60, color="segment",
                           color_discrete_map=SEG_COLORS, barmode="overlay", opacity=0.75,
                           labels={"revenue_at_risk": "Revenue at Risk ($)"})
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        y_col = "clv"
        st.markdown(f"#### CLV vs Churn Probability")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig2 = px.scatter(sample, x="churn_probability", y=y_col,
                          color="segment", color_discrete_map=SEG_COLORS,
                          opacity=0.5, size_max=6,
                          labels={"churn_probability": "Churn Probability", y_col: "CLV ($)"})
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Net profit by segment (BI layer)
    if _has_col(df, "net_profit_per_client"):
        st.markdown("#### Net Profit per Client - Distribution by Segment")
        fig3 = px.box(df, x="segment", y="net_profit_per_client",
                      color="segment", color_discrete_map=SEG_COLORS,
                      labels={"net_profit_per_client": "Net Profit/client ($)", "segment": ""},
                      points=False)
        fig3.add_hline(y=0, line_dash="dash", line_color=C["subtext"],
                       annotation_text="Break-even")
        fig3.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Segment summary table
    st.markdown("#### Segment-Level Financial Breakdown")
    agg_dict = {
        "Customers":       ("customer_id",              "count"),
        "Avg Churn":       ("churn_probability",        "mean"),
        "High Risk":       ("risk_level",               lambda x: (x == "High").sum()),
        "Total Revenue":   ("predicted_revenue_annual", "sum"),
        "Avg CLV":         ("clv",                      "mean"),
        "Total CLV":       ("clv",                      "sum"),
    }
    if _has_col(df, "revenue_at_risk"):
        agg_dict["Revenue at Risk"] = ("revenue_at_risk", "sum")
    if _has_col(df, "net_profit_per_client"):
        agg_dict["Net Profit"] = ("net_profit_per_client", "sum")

    seg_df = df.groupby("segment").agg(**agg_dict).reset_index()
    seg_df["Churn Rate %"] = (seg_df["High Risk"] / seg_df["Customers"] * 100).round(1)
    if "Revenue at Risk" in seg_df.columns:
        seg_df["RAR %"] = (seg_df["Revenue at Risk"] / seg_df["Total Revenue"] * 100).round(1)

    fmt = {"Avg Churn": "{:.2%}", "Total Revenue": "${:,.0f}", "Avg CLV": "${:,.0f}",
           "Total CLV": "${:,.0f}", "Churn Rate %": "{:.1f}%"}
    if "Revenue at Risk" in seg_df.columns:
        fmt["Revenue at Risk"] = "${:,.0f}"
    if "Net Profit" in seg_df.columns:
        fmt["Net Profit"] = "${:,.0f}"

    styled = seg_df.style.format(fmt)
    styled = styled.background_gradient(subset=["Avg Churn"], cmap="Reds")
    styled = styled.background_gradient(subset=["Avg CLV"], cmap="Greens")
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - Revenue Trends  (NEW - time series)
# ══════════════════════════════════════════════════════════════════════════════

def page_revenue_trends(ts: pd.DataFrame | None):
    st.markdown("<h2 style='margin-top:0;'>Revenue Trends - 24-Month Evolution</h2>", unsafe_allow_html=True)

    if ts is None:
        st.warning("Time series data not available. Run `python pipelines/run_bi_pipeline.py` first.")
        return

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_annual = ts["estimated_monthly_revenue"].sum()
    peak_idx     = ts["estimated_monthly_revenue"].idxmax()
    last6_slope  = float(np.polyfit(range(6), ts["estimated_monthly_revenue"].tail(6).values, 1)[0])

    with c1:
        st.metric("Total Revenue (24m)", f"${total_annual:,.0f}")
    with c2:
        st.metric("Avg Monthly Revenue", f"${ts['estimated_monthly_revenue'].mean():,.0f}")
    with c3:
        st.metric("Peak Month", ts["calendar_ym"].iloc[peak_idx],
                  f"${ts['estimated_monthly_revenue'].max():,.0f}")
    with c4:
        direction = "▲ Growing" if last6_slope > 0 else "▼ Declining"
        st.metric("6-Month Trend", direction, f"${last6_slope:+,.0f}/month")

    st.markdown("")

    # ── Dual-axis: Revenue vs At-Risk ─────────────────────────────────────────
    st.markdown("#### Monthly Revenue vs Revenue at Risk")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["calendar_ym"], y=ts["estimated_monthly_revenue"],
        name="Monthly Revenue", line=dict(color=C["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(31,111,235,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=ts["calendar_ym"], y=ts["revenue_3m_rolling_avg"],
        name="3M Rolling Avg", line=dict(color=C["success"], width=1.5, dash="dash"),
    ))
    if "monthly_revenue_at_risk" in ts.columns:
        fig.add_trace(go.Scatter(
            x=ts["calendar_ym"], y=ts["monthly_revenue_at_risk"],
            name="Revenue at Risk", line=dict(color=C["danger"], width=2),
            fill="tozeroy", fillcolor="rgba(218,54,51,0.15)",
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis2=dict(title="Revenue at Risk ($)", overlaying="y", side="right",
                        showgrid=False, color=C["danger"]),
        )
    fig.update_layout(**PLOT_LAYOUT, yaxis_title="Monthly Revenue ($)", xaxis_title="Month")
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # MoM growth
        st.markdown("#### Month-over-Month Revenue Growth (%)")
        if "revenue_mom_growth_pct" in ts.columns:
            ts_clean = ts.dropna(subset=["revenue_mom_growth_pct"])
            colors   = [C["success"] if v >= 0 else C["danger"]
                        for v in ts_clean["revenue_mom_growth_pct"]]
            fig2 = go.Figure(go.Bar(
                x=ts_clean["calendar_ym"], y=ts_clean["revenue_mom_growth_pct"],
                marker_color=colors, name="MoM Growth %",
            ))
            fig2.add_hline(y=0, line_color=C["subtext"], line_width=1)
            fig2.update_layout(**PLOT_LAYOUT, yaxis_title="Growth (%)", xaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        # Transaction volume trend
        st.markdown("#### Monthly Transaction Volume")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=ts["calendar_ym"], y=ts["transaction_volume"],
            marker_color=C["accent"], opacity=0.7, name="Tx Volume",
        ))
        if "tx_volume_3m_rolling_avg" in ts.columns:
            fig3.add_trace(go.Scatter(
                x=ts["calendar_ym"], y=ts["tx_volume_3m_rolling_avg"],
                name="3M Avg", line=dict(color=C["warning"], width=2),
            ))
        fig3.update_layout(**PLOT_LAYOUT, yaxis_title="# Transactions", xaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)

    # Avg churn probability over time
    st.markdown("#### Average Churn Probability of Active Customers (Monthly)")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ts["calendar_ym"], y=ts["avg_churn_probability"] * 100,
        name="Avg Churn %", line=dict(color=C["danger"], width=2.5),
        fill="tozeroy", fillcolor="rgba(218,54,51,0.1)",
    ))
    if "churn_prob_3m_rolling_avg" in ts.columns:
        fig4.add_trace(go.Scatter(
            x=ts["calendar_ym"], y=ts["churn_prob_3m_rolling_avg"] * 100,
            name="3M Avg", line=dict(color=C["warning"], width=1.5, dash="dash"),
        ))
    fig4.update_layout(**PLOT_LAYOUT, yaxis_title="Avg Churn Probability (%)",
                       xaxis_title="Month")
    st.plotly_chart(fig4, use_container_width=True)

    # Raw data table
    with st.expander("View raw time series data"):
        show_cols = [c for c in ["calendar_ym", "transaction_volume",
                                  "estimated_monthly_revenue", "revenue_3m_rolling_avg",
                                  "avg_churn_probability", "monthly_revenue_at_risk",
                                  "revenue_mom_growth_pct", "unique_active_customers"]
                     if c in ts.columns]
        st.dataframe(
            ts[show_cols].style.format({
                "estimated_monthly_revenue":  "${:,.0f}",
                "revenue_3m_rolling_avg":     "${:,.0f}",
                "monthly_revenue_at_risk":    "${:,.0f}",
                "avg_churn_probability":      "{:.2%}",
                "revenue_mom_growth_pct":     "{:+.1f}%",
                "transaction_volume":         "{:,.0f}",
                "unique_active_customers":    "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 - BI Insights  (NEW - engagement, clustering, ROI simulator)
# ══════════════════════════════════════════════════════════════════════════════

def page_bi_insights(df: pd.DataFrame, kpi: dict):
    st.markdown("<h2 style='margin-top:0;'>BI Insights - Behavioral & Profitability Analytics</h2>", unsafe_allow_html=True)

    bi_ready = _has_col(df, "engagement_index")
    if not bi_ready:
        st.warning("Run `python pipelines/run_bi_pipeline.py` to unlock BI Insights.")
        return

    # ── KPI cards from report ─────────────────────────────────────────────────
    kpis = kpi.get("kpis", {})
    eng  = kpis.get("engagement_distribution", {})
    rci  = kpis.get("revenue_concentration", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Engagement Score", f"{eng.get('avg_engagement', 0):.3f}")
    with c2:
        st.metric("Low Engagement (<0.25)", f"{eng.get('low_engagement_count', 0):,}",
                  f"{eng.get('low_engagement_pct', 0):.1f}% of portfolio")
    with c3:
        st.metric("Revenue Concentration",
                  f"{rci.get('top_10pct_revenue_share', 0):.1f}%",
                  "Top 10% customers")
    with c4:
        corr = eng.get("corr_with_churn_prob", 0)
        st.metric("Engagement–Churn Correlation", f"{corr:.3f}",
                  "Negative = lower engagement → higher churn")

    st.markdown("")

    # ── Engagement vs Churn scatter ───────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Engagement Index vs Churn Probability")
        sample = df.sample(min(6000, len(df)), random_state=1)
        fig = px.scatter(
            sample, x="engagement_index", y="churn_probability",
            color="segment", color_discrete_map=SEG_COLORS,
            opacity=0.45,
            labels={"engagement_index": "Engagement Index [0–1]",
                    "churn_probability": "Churn Probability"},
        )
        fig.add_vline(x=0.25, line_dash="dash", line_color=C["warning"],
                      annotation_text="Low engagement threshold")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Spending Trend Slope by Risk Cluster")
        if _has_col(df, "spending_trend_slope") and _has_col(df, "risk_cluster"):
            fig2 = px.box(df, x="risk_cluster", y="spending_trend_slope",
                          color="risk_cluster",
                          color_discrete_map={"low": C["low"],
                                              "medium": C["warning"],
                                              "high": C["high"]},
                          category_orders={"risk_cluster": ["low", "medium", "high"]},
                          labels={"spending_trend_slope": "Monthly Spending Slope ($)",
                                  "risk_cluster": "Risk Cluster"},
                          points=False)
            fig2.add_hline(y=0, line_dash="dash", line_color=C["subtext"],
                           annotation_text="Flat (no trend)")
            fig2.update_layout(**PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Inactivity streak analysis ────────────────────────────────────────────
    if _has_col(df, "inactivity_streak"):
        st.markdown("#### Inactivity Streak - Leading Churn Indicator")
        streak_agg = (
            df.groupby("inactivity_streak")
            .agg(n_customers=("customer_id", "count"),
                 avg_churn=("churn_probability", "mean"))
            .reset_index()
            .query("inactivity_streak <= 12")
        )
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=streak_agg["inactivity_streak"], y=streak_agg["n_customers"],
            name="# Customers", marker_color=C["accent"], opacity=0.7,
        ))
        fig3.add_trace(go.Scatter(
            x=streak_agg["inactivity_streak"],
            y=streak_agg["avg_churn"] * 100,
            name="Avg Churn % (right axis)",
            line=dict(color=C["danger"], width=2.5),
            yaxis="y2",
        ))
        fig3.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="Consecutive Inactive Months",
            yaxis_title="# Customers",
            yaxis2=dict(title="Avg Churn Probability (%)", overlaying="y",
                        side="right", color=C["danger"], showgrid=False),
        )
        fig3.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Risk-Adjusted Value vs Engagement ─────────────────────────────────────
    if _has_col(df, "risk_adjusted_value_score"):
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### Risk-Adjusted Value Score Distribution")
            fig4 = px.histogram(
                df, x="risk_adjusted_value_score", nbins=50,
                color="segment", color_discrete_map=SEG_COLORS,
                barmode="overlay", opacity=0.75,
                labels={"risk_adjusted_value_score": "Risk-Adjusted Value Score [0–1]"},
            )
            fig4.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig4, use_container_width=True)

        with col_d:
            st.markdown("#### Value Cluster Breakdown")
            if _has_col(df, "value_cluster"):
                vc = df.groupby(["value_cluster", "segment"])["customer_id"].count().reset_index(name="count")
                fig5 = px.bar(vc, x="value_cluster", y="count", color="segment",
                              color_discrete_map=SEG_COLORS, barmode="stack",
                              labels={"value_cluster": "CLV Value Cluster",
                                      "count": "# Customers"})
                fig5.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig5, use_container_width=True)

    # ── ROI Simulator ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Retention ROI Simulator")
    st.markdown(f"<span style='color:{C['subtext']};'>Simulate the financial return of your retention programme before committing budget.</span>", unsafe_allow_html=True)

    total_rar = float(df["revenue_at_risk"].sum()) if _has_col(df, "revenue_at_risk") \
        else float((df["predicted_revenue_annual"] * df["churn_probability"]).sum())
    n_high    = int((df["risk_level"] == "High").sum())

    r1, r2 = st.columns(2)
    with r1:
        success_rate = st.slider("Expected retention success rate (%)", 0, 100, 50, 5)
        cost_per_cust = st.number_input("Cost per retention intervention ($)", 0, 5000, 500, 50)
    with r2:
        revenue_saved = total_rar * success_rate / 100
        total_cost    = n_high * cost_per_cust
        net_roi       = revenue_saved - total_cost
        roi_ratio     = revenue_saved / total_cost if total_cost > 0 else 0

        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Revenue Saved", f"${revenue_saved:,.0f}")
        with col_y:
            st.metric("Programme Cost", f"${total_cost:,.0f}")
        with col_z:
            st.metric("Net ROI", f"${net_roi:,.0f}",
                      f"{roi_ratio:.1f}× return on investment")

    # Waterfall
    categories = ["Revenue at Risk", "Revenue Saved", "Programme Cost", "Net ROI"]

    fig_w = go.Figure(go.Waterfall(
        name="", orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=categories,
        y=[total_rar, revenue_saved, -total_cost, 0],
        connector=dict(line=dict(color=C["border"])),
        decreasing=dict(marker_color=C["danger"]),
        increasing=dict(marker_color=C["success"]),
        totals=dict(marker_color=C["accent"]),
        text=[f"${v:,.0f}" for v in [total_rar, revenue_saved, total_cost, net_roi]],
        textposition="outside",
    ))
    fig_w.update_layout(**PLOT_LAYOUT, yaxis_title="Amount ($)")
    st.plotly_chart(fig_w, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 - AI Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def page_ai_recommendations(df: pd.DataFrame):
    st.markdown("<h2 style='margin-top:0;'>AI Recommendations Panel</h2>", unsafe_allow_html=True)

    action_counts = df["recommended_action"].value_counts()
    cols = st.columns(len(action_counts))
    for i, (action, cnt) in enumerate(action_counts.items()):
        with cols[i]:
            pct = cnt / len(df) * 100
            st.metric(action, f"{cnt:,}", f"{pct:.1f}%")

    st.markdown("")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Action Distribution by Segment")
        action_seg = df.groupby(["segment", "recommended_action"]).size().reset_index(name="count")
        fig = px.bar(action_seg, x="segment", y="count", color="recommended_action",
                     color_discrete_map=ACTION_COLORS, barmode="stack",
                     labels={"count": "Customers", "segment": "", "recommended_action": "Action"})
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Priority Score Heatmap")
        heat_df = (
            df.groupby(["risk_level", "financial_value"])["priority_score"]
            .mean().reset_index()
            .pivot(index="risk_level", columns="financial_value", values="priority_score")
        )
        for lvl in ["High", "Medium", "Low"]:
            if lvl not in heat_df.index:   heat_df.loc[lvl] = np.nan
            if lvl not in heat_df.columns: heat_df[lvl] = np.nan
        heat_df = heat_df.reindex(index=["High", "Medium", "Low"],
                                   columns=["High", "Medium", "Low"])
        fig2 = px.imshow(heat_df, color_continuous_scale=[[0, C["success"]], [0.5, C["warning"]], [1, C["danger"]]],
                         text_auto=".3f",
                         labels=dict(x="Financial Value", y="Risk Level", color="Avg Priority"),
                         aspect="auto")
        fig2.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Top priority customers
    st.markdown("#### Top Priority Customers - Immediate Action Required")

    top_cols = ["customer_id", "segment", "churn_probability",
                "predicted_revenue_annual", "clv", "risk_level",
                "financial_value", "recommended_action", "priority_score"]
    if _has_col(df, "revenue_at_risk"):
        top_cols.insert(5, "revenue_at_risk")
    if _has_col(df, "engagement_index"):
        top_cols.append("engagement_index")
    if _has_col(df, "days_since_last_transaction"):
        top_cols.append("days_since_last_transaction")
    if _has_col(df, "action_urgency"):
        top_cols.append("action_urgency")

    top_cols = [c for c in top_cols if c in df.columns]
    top = df[df["risk_level"] == "High"].nlargest(25, "priority_score")[top_cols]

    fmt = {"churn_probability": "{:.1%}", "predicted_revenue_annual": "${:,.0f}",
           "clv": "${:,.0f}", "priority_score": "{:.4f}"}
    if "revenue_at_risk" in top_cols:
        fmt["revenue_at_risk"] = "${:,.0f}"
    if "engagement_index" in top_cols:
        fmt["engagement_index"] = "{:.3f}"
    if "days_since_last_transaction" in top_cols:
        fmt["days_since_last_transaction"] = "{:.0f} days"

    st.dataframe(
        top.style.format(fmt).background_gradient(subset=["churn_probability"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )

    csv = top.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Priority List (CSV)", data=csv,
                       file_name="priority_customers.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 - Model Info
# ══════════════════════════════════════════════════════════════════════════════

def page_model_info(df: pd.DataFrame, metrics: dict, kpi: dict):
    st.markdown("<h2 style='margin-top:0;'>Model & Pipeline Information</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pipeline Architecture")
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| Data Simulation | PySpark (local) / NumPy |
| Storage | Partitioned Parquet (PyArrow) |
| Churn Model | XGBoost Classifier |
| Revenue Model | XGBoost Regressor |
| CLV Scoring | Actuarial (discount model) |
| Decision Engine | Rule-based + ML hybrid |
| BI Feature Layer | Pandas + Scikit-learn KMeans |
| Dashboard | Streamlit + Plotly |
""")
    with col2:
        n = metrics.get("n_customers", len(df))
        st.markdown("#### Simulation Parameters")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Customers | {n:,} |
| Simulation Period | 24 months |
| Segments | retail / premium / high\_net\_worth |
| Discount Rate (CLV) | 10% annual |
| CLV Horizon | 5 years |
| Margin Rate | 18% |
| Churn High Threshold | 65% |
| BI Pipeline Features | 39 columns |
""")

    st.markdown("#### Churn Risk Score Distribution")
    fig = px.histogram(df, x="churn_probability", nbins=50, color="segment",
                       color_discrete_map=SEG_COLORS, opacity=0.75, barmode="overlay",
                       labels={"churn_probability": "Churn Probability"})
    fig.add_vline(x=0.35, line_dash="dash", line_color=C["warning"], annotation_text="Medium threshold")
    fig.add_vline(x=0.65, line_dash="dash", line_color=C["danger"],  annotation_text="High threshold")
    fig.update_layout(**PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    if kpi:
        st.markdown("#### KPI Summary (last BI pipeline run)")
        summ = kpi.get("summary", {})
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Revenue",    f"${_f(summ.get('total_revenue', 0)):,.0f}")
        with s2: st.metric("Revenue at Risk",  f"${_f(summ.get('total_revenue_at_risk', 0)):,.0f}")
        with s3: st.metric("Total 5-yr CLV",   f"${_f(summ.get('total_clv', 0)):,.0f}")
        with s4: st.metric("Net Profit",       f"${_f(summ.get('total_net_profit', 0)):,.0f}")

    if metrics:
        with st.expander("View full business_metrics.json"):
            st.json({k: v for k, v in metrics.items()
                     if k not in ("segment_breakdown", "action_distribution")})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df_raw  = load_customers()
    ts      = load_time_series()
    fa      = load_financial_aggregates()
    kpi     = load_kpi_report()
    metrics = load_metrics()

    if df_raw is None:
        st.error(
            "No pipeline output found. Run the pipeline first:\n\n"
            "```bash\npython pipelines/run_pipeline.py\n"
            "python pipelines/run_bi_pipeline.py\n```"
        )
        return

    df = render_sidebar(df_raw)

    if df.empty:
        st.warning("No customers match current filters.")
        return

    bi_active = _has_col(df, "engagement_index")
    st.markdown(
        f"<div style='background:{C['surface']}; border:1px solid {C['border']}; "
        f"border-radius:10px; padding:12px 20px; margin-bottom:16px;'>"
        f"<span style='font-size:1.1rem; font-weight:600;'>🏦 CBSFIM &nbsp;·&nbsp; "
        f"Customer Behavior & Strategic Financial Impact Modeling</span>"
        f"<span style='float:right; color:{C['subtext']}; font-size:0.8rem; margin-top:4px;'>"
        f"{len(df):,} customers"
        + ("&nbsp;·&nbsp;<span style='color:#3FB950;'>BI Layer ✓</span>" if bi_active else "")
        + "&nbsp;·&nbsp; v2.0</span></div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📊 Executive Summary",
        "🔍 Customer Risk Explorer",
        "💰 Financial Impact",
        "📈 Revenue Trends",
        "💡 BI Insights",
        "🤖 AI Recommendations",
        "⚙️ Model Info",
    ])

    with tabs[0]: page_executive_summary(df, kpi)
    with tabs[1]: page_customer_risk_explorer(df)
    with tabs[2]: page_financial_impact(df, fa)
    with tabs[3]: page_revenue_trends(ts)
    with tabs[4]: page_bi_insights(df, kpi)
    with tabs[5]: page_ai_recommendations(df)
    with tabs[6]: page_model_info(df, metrics, kpi)

    st.markdown(
        f"<div style='text-align:center; color:{C['subtext']}; font-size:0.73rem; "
        f"padding:20px 0 8px 0; border-top:1px solid {C['border']}; margin-top:24px;'>"
        f"CBSFIM Platform - Big Data + AI + BI Decision System"
        f"<br><span style='color:{C['text']}; font-weight:600;'>Realise par Oswald Jaures KOFFI</span>"
        f"&nbsp;·&nbsp; Donnees synthetiques - a des fins de demonstration"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
