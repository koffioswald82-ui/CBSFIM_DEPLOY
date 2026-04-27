"""
Business Metrics Computation
-----------------------------
Computes executive-level financial KPIs from the recommendations table:

  - revenue_at_risk         : revenue threatened by high-churn customers
  - total_expected_loss     : probability-weighted revenue loss
  - clv_at_risk             : CLV destruction from churn
  - high_value_at_risk_count: # high-value customers with High risk
  - segment_breakdown       : per-segment summary
  - churn_rate              : % customers classified as High risk
"""

import pandas as pd
import logging
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)


def compute_metrics(recommendations: pd.DataFrame) -> dict:
    df = recommendations.copy()
    n  = len(df)

    # ── Global KPIs ────────────────────────────────────────────────────────────
    high_risk       = df[df["risk_level"] == "High"]
    churn_rate      = len(high_risk) / n

    revenue_at_risk = high_risk["predicted_revenue_annual"].sum()

    # Expected loss: for every customer, loss = revenue * churn_probability
    df["expected_loss"] = df["predicted_revenue_annual"] * df["churn_probability"]
    total_expected_loss = df["expected_loss"].sum()

    clv_at_risk     = high_risk["clv"].sum()
    high_value_at_risk = len(
        df[(df["risk_level"] == "High") & (df["financial_value"] == "High")]
    )

    total_revenue   = df["predicted_revenue_annual"].sum()
    total_clv       = df["clv"].sum()

    # ── Segment breakdown ──────────────────────────────────────────────────────
    seg_agg = (
        df.groupby("segment")
        .agg(
            n_customers        = ("customer_id",              "count"),
            avg_churn_prob     = ("churn_probability",        "mean"),
            high_risk_count    = ("risk_level",               lambda x: (x == "High").sum()),
            revenue_at_risk_seg= ("predicted_revenue_annual", lambda x: x[df.loc[x.index, "risk_level"] == "High"].sum()),
            avg_clv            = ("clv",                      "mean"),
            total_revenue_seg  = ("predicted_revenue_annual", "sum"),
        )
        .reset_index()
        .round(2)
    )
    seg_agg["churn_rate_pct"] = (seg_agg["high_risk_count"] / seg_agg["n_customers"] * 100).round(1)

    # ── Action distribution ───────────────────────────────────────────────────
    action_dist = df["recommended_action"].value_counts().to_dict()

    metrics = {
        "n_customers":          n,
        "churn_rate":           round(churn_rate, 4),
        "churn_rate_pct":       round(churn_rate * 100, 2),
        "revenue_at_risk":      round(revenue_at_risk, 2),
        "total_expected_loss":  round(total_expected_loss, 2),
        "clv_at_risk":          round(clv_at_risk, 2),
        "high_value_at_risk":   int(high_value_at_risk),
        "total_revenue":        round(total_revenue, 2),
        "total_clv":            round(total_clv, 2),
        "revenue_at_risk_pct":  round(revenue_at_risk / total_revenue * 100, 2) if total_revenue else 0,
        "action_distribution":  action_dist,
        "segment_breakdown":    seg_agg.to_dict(orient="records"),
    }

    # Persist
    out_json    = PROCESSED_DIR / "business_metrics.json"
    out_parquet = PROCESSED_DIR / "segment_breakdown.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    seg_agg.to_parquet(out_parquet, index=False)

    logger.info(
        "Business metrics computed:\n"
        "  Customers        : %d\n"
        "  Churn rate       : %.2f%%\n"
        "  Revenue at Risk  : $%s\n"
        "  Expected Loss    : $%s\n"
        "  High-value @ Risk: %d",
        n, churn_rate * 100,
        f"{revenue_at_risk:,.0f}", f"{total_expected_loss:,.0f}",
        high_value_at_risk
    )
    return metrics


def load_metrics() -> dict:
    path = PROCESSED_DIR / "business_metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
