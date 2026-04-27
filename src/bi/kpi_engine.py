"""
KPI Engine
-----------
Computes every executive KPI with its business definition, formula,
computed value, and strategic implication.  Output is a structured dict
persisted as kpi_report.json for downstream dashboards and audit trails.

KPIs defined:
  1. Revenue at Risk (RAR)
  2. Churn Rate (CR)
  3. Customer Lifetime Value (CLV)
  4. High-Value Customers at Risk (HVAR)
  5. Net Profit per Segment (NPS_SEG)
  6. Engagement Score Distribution (ESD)
  7. Revenue Concentration Index (RCI)
  8. Time-Series KPIs (revenue trend, growth slope)
"""

import logging
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import REVENUE_MARGIN_RATE, CLV_DISCOUNT_RATE, CLV_YEARS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_kpis(
    enriched:   pd.DataFrame,
    timeseries: pd.DataFrame,
) -> dict:
    """
    Computes all KPIs and returns a fully documented report dict.
    Persists to kpi_report.json via the caller (run_bi_pipeline).
    """
    logger.info("  Computing executive KPI report...")

    df = enriched.copy()
    n  = len(df)

    total_revenue         = df["predicted_revenue_annual"].sum()
    total_clv             = df["clv"].sum()
    total_revenue_at_risk = df["revenue_at_risk"].sum()
    total_net_profit      = df["net_profit_per_client"].sum()

    high_risk = df[df["risk_level"] == "High"]
    critical  = df[(df["risk_level"] == "High") & (df["financial_value"] == "High")]

    # ── 1. Revenue at Risk ────────────────────────────────────────────────────
    rar_by_segment = (
        df.groupby("segment")["revenue_at_risk"].sum().round(2).to_dict()
    )
    rar_by_region = (
        df.groupby("region")["revenue_at_risk"].sum().round(2).to_dict()
    )

    kpi_rar = {
        "name":  "Revenue at Risk",
        "code":  "RAR",
        "formula": (
            "SUM(predicted_revenue_annual × churn_probability)"
            " — for every customer"
        ),
        "business_definition": (
            "Probability-weighted annual revenue expected to be lost if "
            "at-risk customers churn.  Unlike a simple binary high/low split, "
            "RAR uses each customer's individual churn probability to produce "
            "a statistically grounded loss estimate."
        ),
        "value":                   round(total_revenue_at_risk, 2),
        "value_formatted":         f"${total_revenue_at_risk:,.0f}",
        "as_pct_of_total_revenue": round(total_revenue_at_risk / total_revenue * 100, 2),
        "by_segment":              rar_by_segment,
        "by_region":               rar_by_region,
        "strategic_implication": (
            "A RAR above 5 % of total revenue warrants a board-level retention "
            "programme.  Allocate intervention budget proportional to RAR by "
            "segment — never spend more than 30 % of a customer's CLV on "
            "a single retention offer."
        ),
    }

    # ── 2. Churn Rate ─────────────────────────────────────────────────────────
    churn_rate_pct = len(high_risk) / n * 100
    churn_by_seg   = (
        df.groupby("segment")
        .apply(lambda x: round(len(x[x["risk_level"] == "High"]) / len(x) * 100, 1))
        .to_dict()
    )

    kpi_churn = {
        "name":  "Churn Rate",
        "code":  "CR",
        "formula": (
            "COUNT(customers WHERE risk_level = 'High') / COUNT(all_customers) × 100"
        ),
        "business_definition": (
            "Percentage of the customer base classified as high churn risk by "
            "the XGBoost churn model.  In retail banking, sustainable churn "
            "rates are typically 8–12 %.  Above 20 % signals systemic "
            "service-quality or competitive-pricing failure."
        ),
        "value":                round(churn_rate_pct, 2),
        "value_formatted":      f"{churn_rate_pct:.1f}%",
        "by_segment":           churn_by_seg,
        "expected_annual_loss_count": round(df["churn_probability"].mean() * n),
        "avg_churn_probability": round(df["churn_probability"].mean(), 4),
        "strategic_implication": (
            "Track monthly.  A rise of >2 pp per quarter calls for an "
            "immediate competitive and service-quality audit.  The retail "
            "segment typically drives aggregate churn — prioritise digital "
            "engagement programmes there first."
        ),
    }

    # ── 3. Customer Lifetime Value ────────────────────────────────────────────
    discount_factor = sum(
        1 / (1 + CLV_DISCOUNT_RATE) ** t for t in range(1, CLV_YEARS + 1)
    )
    clv_by_seg = (
        df.groupby("segment")["clv"]
        .agg(mean="mean", total="sum", median="median")
        .round(2)
        .to_dict()
    )

    kpi_clv = {
        "name":  "Customer Lifetime Value",
        "code":  "CLV",
        "formula": (
            f"predicted_revenue_annual × {REVENUE_MARGIN_RATE} "
            f"× (1 − churn_probability) × Σ[1/(1+{CLV_DISCOUNT_RATE})^t]"
            f" for t = 1 … {CLV_YEARS}"
        ),
        "discount_factor": round(discount_factor, 4),
        "margin_rate":     REVENUE_MARGIN_RATE,
        "horizon_years":   CLV_YEARS,
        "business_definition": (
            f"Net present value of the expected net cash flows from a customer "
            f"over {CLV_YEARS} years, discounted at "
            f"{CLV_DISCOUNT_RATE * 100:.0f} % and adjusted for the probability "
            "that the customer stays.  CLV sets the ceiling on economically "
            "rational retention spend."
        ),
        "total_clv":            round(total_clv, 2),
        "total_clv_formatted":  f"${total_clv:,.0f}",
        "avg_clv":              round(df["clv"].mean(), 2),
        "median_clv":           round(df["clv"].median(), 2),
        "p90_clv":              round(df["clv"].quantile(0.90), 2),
        "clv_at_risk":          round(high_risk["clv"].sum(), 2),
        "by_segment":           clv_by_seg,
        "strategic_implication": (
            "Never invest more than CLV × 0.30 on a single retention "
            "intervention.  Use CLV tiers to stratify offer value: "
            "Elite (top 10 %) → premium concierge, Preferred → digital "
            "incentive, Standard → automated nudge."
        ),
    }

    # ── 4. High-Value Customers at Risk ───────────────────────────────────────
    hvar_by_seg = (
        critical.groupby("segment")["customer_id"].count().to_dict()
    )

    kpi_hvar = {
        "name":  "High-Value Customers at Risk",
        "code":  "HVAR",
        "formula": (
            "COUNT(customers WHERE risk_level = 'High' AND financial_value = 'High')"
        ),
        "business_definition": (
            "Customers who simultaneously belong to the top revenue quartile "
            "AND carry high churn risk.  This is the highest-urgency cohort — "
            "their departure would cause disproportionate revenue destruction."
        ),
        "count":                int(len(critical)),
        "revenue_at_stake":     round(critical["predicted_revenue_annual"].sum(), 2),
        "clv_at_stake":         round(critical["clv"].sum(), 2),
        "avg_churn_probability": round(critical["churn_probability"].mean(), 4),
        "pct_of_total_customers": round(len(critical) / n * 100, 2),
        "pct_of_total_revenue": round(
            critical["predicted_revenue_annual"].sum() / total_revenue * 100, 2
        ),
        "by_segment":           hvar_by_seg,
        "strategic_implication": (
            "Mandate personal outreach within 48 hours — not automated email. "
            "Assign a named relationship manager to each HVAR customer. "
            "Even retaining 50 % of this cohort preserves a "
            "disproportionate share of lifetime revenue."
        ),
    }

    # ── 5. Net Profit per Segment ─────────────────────────────────────────────
    seg_profit = (
        df.groupby("segment")
        .agg(
            total_net_profit  = ("net_profit_per_client",    "sum"),
            avg_net_profit    = ("net_profit_per_client",    "mean"),
            total_margin      = ("estimated_margin_per_client", "sum"),
            total_cost        = ("cost_to_serve",            "sum"),
            n_customers       = ("customer_id",              "count"),
        )
        .round(2)
        .to_dict()
    )

    kpi_profit = {
        "name":  "Net Profit per Segment",
        "code":  "NPS_SEG",
        "formula": "SUM(estimated_margin_per_client − cost_to_serve)  grouped by segment",
        "business_definition": (
            "Segment-level net profitability after deducting the annual cost "
            "to serve each customer.  Reveals cross-subsidisation dynamics "
            "and where to focus operational efficiency programmes."
        ),
        "total_net_profit":           round(total_net_profit, 2),
        "total_net_profit_formatted": f"${total_net_profit:,.0f}",
        "overall_profit_margin_pct":  round(total_net_profit / total_revenue * 100, 2),
        "by_segment":                 seg_profit,
        "strategic_implication": (
            "If the retail segment delivers negative net profit, the bank must "
            "either increase fee income, digitise servicing (to lower cost-to-"
            "serve below $180), or selectively reprice the segment.  HNW "
            "profit subsidises the mass-market digital strategy — protect it."
        ),
    }

    # ── 6. Engagement Score Distribution ─────────────────────────────────────
    eng_by_seg   = df.groupby("segment")["engagement_index"].mean().round(4).to_dict()
    low_eng_mask = df["engagement_index"] < 0.25
    corr_churn   = df["engagement_index"].corr(df["churn_probability"])

    kpi_engagement = {
        "name":  "Engagement Score Distribution",
        "code":  "ESD",
        "formula": (
            "0.30 × active_days_ratio"
            " + 0.30 × norm(tx_count_90d)"
            " + 0.20 × online_channel_ratio"
            " + 0.20 × recency_score"
        ),
        "business_definition": (
            "Composite behavioural score [0, 1] measuring how actively a "
            "customer uses the bank's services.  Low engagement is the most "
            "reliable leading indicator of churn — typically 3-6 months before "
            "account closure."
        ),
        "avg_engagement":         round(df["engagement_index"].mean(), 4),
        "low_engagement_count":   int(low_eng_mask.sum()),
        "low_engagement_pct":     round(low_eng_mask.mean() * 100, 2),
        "corr_with_churn_prob":   round(corr_churn, 4),
        "by_segment":             eng_by_seg,
        "strategic_implication": (
            "Launch digital activation campaigns targeting engagement < 0.25. "
            "Goal: lift 20 % of low-engagement customers to medium tier within "
            "6 months via push notifications, personalised offers, and "
            "onboarding nudges.  Track weekly — engagement is a leading KPI."
        ),
    }

    # ── 7. Revenue Concentration Index ───────────────────────────────────────
    sorted_rev   = df["predicted_revenue_annual"].sort_values(ascending=False)
    top10_n      = max(1, int(n * 0.10))
    top10_rev    = sorted_rev.iloc[:top10_n].sum()
    top20_n      = max(1, int(n * 0.20))
    top20_rev    = sorted_rev.iloc[:top20_n].sum()

    kpi_concentration = {
        "name":  "Revenue Concentration Index",
        "code":  "RCI",
        "formula": "SUM(revenue of top 10 % customers) / SUM(all revenue)",
        "business_definition": (
            "Share of total annual revenue generated by the highest-value "
            "10 % of customers.  A Pareto-type measure of portfolio "
            "concentration risk.  Above 40 % indicates key-client dependency."
        ),
        "top_10pct_revenue_share":  round(top10_rev / total_revenue * 100, 2),
        "top_10pct_revenue_amount": round(top10_rev, 2),
        "top_20pct_revenue_share":  round(top20_rev / total_revenue * 100, 2),
        "top_10pct_customer_count": top10_n,
        "strategic_implication": (
            "If top 10 % generates >50 % of revenue, losing even 5 % of that "
            "cohort erodes the bottom line materially.  Mandate quarterly "
            "executive-level reviews for all customers in the top revenue "
            "decile regardless of their current churn risk classification."
        ),
    }

    # ── 8. Time-Series KPIs ───────────────────────────────────────────────────
    kpi_ts: dict = {}
    if len(timeseries) >= 2:
        ts_rev = timeseries["estimated_monthly_revenue"]
        last6  = ts_rev.tail(6)
        slope  = float(np.polyfit(range(len(last6)), last6.values, 1)[0])
        peak_idx = ts_rev.idxmax()

        kpi_ts = {
            "latest_month":          str(timeseries["calendar_ym"].iloc[-1]),
            "latest_monthly_revenue":round(float(ts_rev.iloc[-1]), 2),
            "revenue_trend_6m_slope":round(slope, 2),
            "revenue_trend_direction": "GROWING" if slope > 0 else "DECLINING",
            "peak_revenue_month":    str(timeseries["calendar_ym"].iloc[peak_idx]),
            "peak_monthly_revenue":  round(float(ts_rev.max()), 2),
            "avg_monthly_revenue":   round(float(ts_rev.mean()), 2),
            "avg_monthly_tx_volume": round(
                float(timeseries["transaction_volume"].mean()), 0
            ),
        }

    # ── Assemble report ───────────────────────────────────────────────────────
    report = {
        "metadata": {
            "generated_at":       pd.Timestamp.now().isoformat(),
            "n_customers":        n,
            "observation_period": "Jan 2023 – Dec 2024",
            "currency":           "USD",
            "margin_rate":        REVENUE_MARGIN_RATE,
            "clv_discount_rate":  CLV_DISCOUNT_RATE,
            "clv_horizon_years":  CLV_YEARS,
        },
        "summary": {
            "total_revenue":         round(total_revenue, 2),
            "total_revenue_at_risk": round(total_revenue_at_risk, 2),
            "rar_pct":               round(total_revenue_at_risk / total_revenue * 100, 2),
            "churn_rate_pct":        round(churn_rate_pct, 2),
            "total_clv":             round(total_clv, 2),
            "total_net_profit":      round(total_net_profit, 2),
            "critical_customers":    int(len(critical)),
        },
        "kpis": {
            "revenue_at_risk":         kpi_rar,
            "churn_rate":              kpi_churn,
            "customer_lifetime_value": kpi_clv,
            "high_value_at_risk":      kpi_hvar,
            "net_profit_per_segment":  kpi_profit,
            "engagement_distribution": kpi_engagement,
            "revenue_concentration":   kpi_concentration,
            "time_series":             kpi_ts,
        },
    }

    logger.info(
        "  KPI report complete — RAR: $%s (%.1f%%) | Churn: %.1f%% | CLV: $%s",
        f"{total_revenue_at_risk:,.0f}",
        total_revenue_at_risk / total_revenue * 100,
        churn_rate_pct,
        f"{total_clv:,.0f}",
    )
    return report
