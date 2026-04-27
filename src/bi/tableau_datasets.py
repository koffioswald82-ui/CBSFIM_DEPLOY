"""
Tableau Dataset Builder
------------------------
Produces three analytical tables optimised for Tableau Live Connect:

  A. Customer Table       — granular, 200 k rows, one row per customer
  B. Financial Aggregates — multi-dimensional segment/region/risk summaries
  C. Time Series Table    — 24 monthly snapshots for trend dashboards

Both CSV (utf-8-sig, Tableau-safe) and Parquet outputs are generated.
Column names use lowercase_underscores throughout to match Tableau conventions.
"""

import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

SIM_START = pd.Timestamp("2023-01-01")
N_MONTHS  = 24

# ── Display mapping constants ─────────────────────────────────────────────────
SEGMENT_LABEL = {
    "retail":         "Retail Banking",
    "premium":        "Premium Banking",
    "high_net_worth": "High Net Worth",
}

ACTION_URGENCY = {
    "Retain - Premium Offer":  "CRITICAL",
    "Retain - Standard Offer": "HIGH",
    "Cross-Sell Opportunity":  "MEDIUM",
    "Monitor":                 "LOW",
    "No Action":               "MINIMAL",
}

ACTION_CODE = {
    "Retain - Premium Offer":  "RETAIN_PREMIUM",
    "Retain - Standard Offer": "RETAIN_STANDARD",
    "Cross-Sell Opportunity":  "CROSS_SELL",
    "Monitor":                 "MONITOR",
    "No Action":               "NO_ACTION",
}


# ══════════════════════════════════════════════════════════════════════════════
# A. CUSTOMER TABLE  (granular)
# ══════════════════════════════════════════════════════════════════════════════

def build_customer_table(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Selects, renames, and enriches columns for the customer-level Tableau table.
    Adds display-friendly categorical columns (segment_label, value_tier,
    age_band, action_urgency) and binary flags for easy Tableau filtering.
    """
    logger.info("  Building Customer Table (%d rows)...", len(enriched))

    # Ordered column specification: source_name → output_name
    col_map = {
        # ── Identity ──────────────────────────────────────────────────────────
        "customer_id":                 "customer_id",
        "segment":                     "segment",
        "region":                      "region",
        "age":                         "age",
        "income":                      "income",
        "account_balance":             "account_balance",
        "customer_tenure_years":       "customer_tenure_years",
        # ── Behavior ─────────────────────────────────────────────────────────
        "transaction_count_30d":       "transaction_count_30d",
        "transaction_count_90d":       "transaction_count_90d",
        "active_days_ratio":           "active_days_ratio",
        "spending_trend_slope":        "spending_trend_slope",
        "spending_volatility":         "spending_volatility",
        "avg_transaction_amount":      "avg_transaction_amount",
        "engagement_index":            "engagement_index",
        # ── Churn signals ─────────────────────────────────────────────────────
        "days_since_last_transaction": "days_since_last_transaction",
        "inactivity_streak":           "inactivity_streak",
        "balance_decay_rate":          "balance_decay_rate",
        # ── Financial ────────────────────────────────────────────────────────
        "total_revenue_generated":     "total_revenue_generated",
        "estimated_margin_per_client": "estimated_margin_per_client",
        "cost_to_serve":               "cost_to_serve",
        "net_profit_per_client":       "net_profit_per_client",
        "clv":                         "clv_5yr",
        # ── Risk / ML ────────────────────────────────────────────────────────
        "churn_probability":           "churn_probability",
        "churn_probability_pct":       "churn_probability_pct",
        "revenue_at_risk":             "revenue_at_risk",
        "risk_level":                  "risk_level",
        "risk_cluster":                "risk_cluster",
        "value_cluster":               "value_cluster",
        "financial_value":             "financial_value",
        "financial_value_score":       "financial_value_score",
        "customer_value_score":        "customer_value_score",
        "risk_adjusted_value_score":   "risk_adjusted_value_score",
        # ── Actions ───────────────────────────────────────────────────────────
        "recommended_action":          "recommended_action",
        "priority_score":              "priority_score",
    }

    available = {k: v for k, v in col_map.items() if k in enriched.columns}
    out = enriched[list(available.keys())].rename(columns=available).copy()

    # ── Derived display columns ────────────────────────────────────────────────
    out["segment_label"] = out["segment"].map(SEGMENT_LABEL).fillna(out["segment"])
    out["action_urgency"] = out["recommended_action"].map(ACTION_URGENCY).fillna("LOW")
    out["action_code"]    = out["recommended_action"].map(ACTION_CODE).fillna("MONITOR")

    # Value tier for heatmap colour-coding (Elite → Standard)
    p90 = out["clv_5yr"].quantile(0.90)
    p70 = out["clv_5yr"].quantile(0.70)
    p40 = out["clv_5yr"].quantile(0.40)
    out["value_tier"] = pd.cut(
        out["clv_5yr"],
        bins=[-np.inf, p40, p70, p90, np.inf],
        labels=["Standard", "Preferred", "Premium", "Elite"],
    ).astype(str)

    # Age band for demographic drill-down
    out["age_band"] = pd.cut(
        out["age"],
        bins=[17, 29, 39, 49, 59, 75],
        labels=["18-29", "30-39", "40-49", "50-59", "60+"],
    ).astype(str)

    # Binary flags — makes Tableau calculated fields trivial
    out["is_high_risk"]  = (out["risk_level"] == "High").astype(int)
    out["is_high_value"] = (out["financial_value"] == "High").astype(int)
    out["is_critical"]   = (out["is_high_risk"] & out["is_high_value"]).astype(int)

    logger.info(
        "  Customer Table ready: %d rows × %d columns", len(out), len(out.columns)
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# B. FINANCIAL AGGREGATES TABLE  (multi-dimensional summary)
# ══════════════════════════════════════════════════════════════════════════════

def build_financial_aggregates(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates financial KPIs along six grouping dimensions so Tableau can
    drive executive scorecards, segment bars, and region heatmaps from a
    single data source using the aggregation_level filter.

    Dimensions: total | segment | region | risk_level | action | segment×risk
                segment×region
    """
    logger.info("  Building Financial Aggregates Table...")

    df = enriched.copy()
    # Precompute binary flags for aggregation lambdas
    df["_is_high_risk"]  = (df["risk_level"] == "High")
    df["_is_high_value"] = (df["financial_value"] == "High")
    df["_is_critical"]   = df["_is_high_risk"] & df["_is_high_value"]

    def _agg(group_cols, level_label: str) -> pd.DataFrame:
        """Aggregate all KPI metrics for the given group dimensions."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        agg = df.groupby(group_cols, observed=True).agg(
            n_customers              = ("customer_id",              "count"),
            total_revenue            = ("predicted_revenue_annual", "sum"),
            avg_revenue_per_customer = ("predicted_revenue_annual", "mean"),
            total_revenue_at_risk    = ("revenue_at_risk",          "sum"),
            total_net_profit         = ("net_profit_per_client",    "sum"),
            avg_net_profit           = ("net_profit_per_client",    "mean"),
            total_clv                = ("clv",                      "sum"),
            avg_clv                  = ("clv",                      "mean"),
            avg_churn_probability    = ("churn_probability",        "mean"),
            high_risk_count          = ("_is_high_risk",            "sum"),
            high_value_count         = ("_is_high_value",           "sum"),
            critical_count           = ("_is_critical",             "sum"),
            avg_engagement           = ("engagement_index",         "mean"),
            avg_priority_score       = ("priority_score",           "mean"),
            total_cost_to_serve      = ("cost_to_serve",            "sum"),
        ).reset_index()

        agg["churn_rate_pct"]        = (agg["high_risk_count"] / agg["n_customers"] * 100).round(2)
        agg["revenue_at_risk_pct"]   = (agg["total_revenue_at_risk"] / agg["total_revenue"] * 100).round(2)
        agg["profitability_ratio"]   = (agg["total_net_profit"] / agg["total_revenue"] * 100).round(2)
        agg["aggregation_level"]     = level_label

        # Flatten group key into a single dimension_key column for Tableau pivots
        agg["dimension_key"] = agg[group_cols].astype(str).agg(" | ".join, axis=1)

        return agg.round(2)

    # ── Overall totals ─────────────────────────────────────────────────────────
    total_rev = df["predicted_revenue_annual"].sum()
    total = pd.DataFrame([{
        "dimension_key":         "ALL",
        "aggregation_level":     "total",
        "n_customers":           len(df),
        "total_revenue":         round(total_rev, 2),
        "avg_revenue_per_customer": round(df["predicted_revenue_annual"].mean(), 2),
        "total_revenue_at_risk": round(df["revenue_at_risk"].sum(), 2),
        "total_net_profit":      round(df["net_profit_per_client"].sum(), 2),
        "avg_net_profit":        round(df["net_profit_per_client"].mean(), 2),
        "total_clv":             round(df["clv"].sum(), 2),
        "avg_clv":               round(df["clv"].mean(), 2),
        "avg_churn_probability": round(df["churn_probability"].mean(), 4),
        "high_risk_count":       int(df["_is_high_risk"].sum()),
        "high_value_count":      int(df["_is_high_value"].sum()),
        "critical_count":        int(df["_is_critical"].sum()),
        "avg_engagement":        round(df["engagement_index"].mean(), 4),
        "avg_priority_score":    round(df["priority_score"].mean(), 4),
        "total_cost_to_serve":   round(df["cost_to_serve"].sum(), 2),
        "churn_rate_pct":        round(df["_is_high_risk"].mean() * 100, 2),
        "revenue_at_risk_pct":   round(df["revenue_at_risk"].sum() / total_rev * 100, 2),
        "profitability_ratio":   round(df["net_profit_per_client"].sum() / total_rev * 100, 2),
    }])

    combined = pd.concat([
        total,
        _agg("segment",                         "by_segment"),
        _agg("region",                           "by_region"),
        _agg("risk_level",                       "by_risk_level"),
        _agg("recommended_action",               "by_action"),
        _agg(["segment", "risk_level"],          "by_segment_risk"),
        _agg(["segment", "region"],              "by_segment_region"),
        _agg(["risk_level", "recommended_action"],"by_risk_action"),
    ], ignore_index=True)

    combined = combined.round(2)
    logger.info("  Financial Aggregates ready: %d rows", len(combined))
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# C. TIME SERIES TABLE  (24 monthly snapshots)
# ══════════════════════════════════════════════════════════════════════════════

def build_time_series(
    transactions: pd.DataFrame,
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates transactions and ML risk scores into 24 monthly snapshots.
    Covers Jan 2023 – Dec 2024 (simulation period).
    Includes 3-month rolling averages and MoM growth rates for Tableau
    trend line visualisations.
    """
    logger.info("  Building Time Series Table (24 months)...")

    tx = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(tx["timestamp"]):
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    # Map simulation month integer (0-23) → calendar year-month string "YYYY-MM"
    def _month_to_ym(m: int) -> str:
        dt = SIM_START + pd.DateOffset(months=int(m))
        return dt.strftime("%Y-%m")

    tx["calendar_ym"] = tx["month"].apply(_month_to_ym)

    # ── Core transaction metrics per month ────────────────────────────────────
    monthly_tx = (
        tx.groupby("calendar_ym")
        .agg(
            transaction_volume        = ("transaction_id", "count"),
            total_transaction_amount  = ("amount",         "sum"),
            avg_transaction_amount    = ("amount",         "mean"),
            unique_active_customers   = ("customer_id",    "nunique"),
        )
        .reset_index()
    )

    # Estimated monthly revenue: annualised margin allocated per month
    # total_tx_amount × margin_rate / 24 months
    monthly_tx["estimated_monthly_revenue"] = (
        monthly_tx["total_transaction_amount"] * 0.18 / 24
    ).round(2)

    # ── Risk metrics: average churn probability of customers active that month ─
    cust_risk = enriched[["customer_id", "churn_probability",
                           "revenue_at_risk", "risk_level"]].copy()
    tx_risk = tx[["customer_id", "calendar_ym"]].merge(cust_risk, on="customer_id", how="left")

    monthly_risk = (
        tx_risk.groupby("calendar_ym")
        .agg(
            avg_churn_probability   = ("churn_probability", "mean"),
            high_risk_active_count  = ("risk_level",        lambda x: (x == "High").sum()),
        )
        .reset_index()
    )

    # Monthly revenue-at-risk: distribute annual RAR evenly across 24 months
    monthly_risk["monthly_revenue_at_risk"] = (
        tx_risk.groupby("calendar_ym")["revenue_at_risk"].sum() / 24
    ).reset_index(drop=True)

    # ── Per-segment monthly revenue breakdown ─────────────────────────────────
    seg_monthly = (
        tx.groupby(["calendar_ym", "segment"])["amount"]
        .sum()
        .unstack(fill_value=0)
    )
    seg_monthly.columns = [f"tx_amount_{c}" for c in seg_monthly.columns]
    seg_monthly = seg_monthly.reset_index()

    # ── Merge and sort ─────────────────────────────────────────────────────────
    ts = monthly_tx.merge(monthly_risk, on="calendar_ym", how="left")
    ts = ts.merge(seg_monthly, on="calendar_ym", how="left")
    ts = ts.sort_values("calendar_ym").reset_index(drop=True)

    # Calendar date column for Tableau date axis
    ts["calendar_date"] = pd.to_datetime(ts["calendar_ym"] + "-01")
    ts["month_index"]   = range(1, len(ts) + 1)  # 1–24

    # ── Smoothed trend indicators ──────────────────────────────────────────────
    ts["revenue_3m_rolling_avg"] = (
        ts["estimated_monthly_revenue"].rolling(3, min_periods=1).mean().round(2)
    )
    ts["churn_prob_3m_rolling_avg"] = (
        ts["avg_churn_probability"].rolling(3, min_periods=1).mean().round(4)
    )
    ts["tx_volume_3m_rolling_avg"] = (
        ts["transaction_volume"].rolling(3, min_periods=1).mean().round(0)
    )

    # Month-over-month revenue growth
    ts["revenue_mom_growth_pct"] = (
        ts["estimated_monthly_revenue"].pct_change().mul(100).round(2)
    )

    ts = ts.round(4)
    logger.info("  Time Series Table ready: %d months", len(ts))
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_all(
    customer_df:    pd.DataFrame,
    financial_df:   pd.DataFrame,
    timeseries_df:  pd.DataFrame,
    output_dir:     Path,
) -> dict[str, str]:
    """
    Exports all three datasets as both CSV and Parquet.

    CSV  — utf-8-sig encoding so Excel and Tableau open without BOM issues.
    Parquet — columnar format; load directly into Tableau Hyper via TabPy
              or connect via Tableau's built-in Parquet connector.

    Returns a dict mapping {dataset_csv, dataset_parquet} → file path strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "customer_table":       customer_df,
        "financial_aggregates": financial_df,
        "time_series":          timeseries_df,
    }

    paths: dict[str, str] = {}

    for name, df in datasets.items():
        # Convert categorical columns to string for maximum compatibility
        df = df.copy()
        for col in df.select_dtypes(include="category").columns:
            df[col] = df[col].astype(str)

        csv_path     = output_dir / f"{name}.csv"
        parquet_path = output_dir / f"{name}.parquet"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_parquet(parquet_path, index=False)

        paths[f"{name}_csv"]     = str(csv_path)
        paths[f"{name}_parquet"] = str(parquet_path)

        logger.info(
            "  Exported %-25s → %d rows × %d cols  [CSV + Parquet]",
            name, len(df), len(df.columns),
        )

    return paths
