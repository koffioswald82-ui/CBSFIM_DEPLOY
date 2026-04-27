"""
AI Strategic Recommendation Engine
------------------------------------
Combines ML predictions + business rules to produce per-customer:
  - risk_level       : Low | Medium | High
  - financial_value  : Low | Medium | High
  - recommended_action
  - action_rationale  (explainable)
  - priority_score   (float — for sorting dashboards)

Decision matrix
---------------
              | Value HIGH     | Value MEDIUM    | Value LOW
Risk HIGH     | Retain Premium | Retain Standard | Monitor
Risk MEDIUM   | Cross-Sell     | Monitor         | No Action
Risk LOW      | Cross-Sell     | No Action       | No Action
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    REVENUE_PERCENTILE_HIGH, REVENUE_PERCENTILE_LOW, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# ── Thresholds (computed dynamically from data) ────────────────────────────────
_VALUE_P_HIGH = REVENUE_PERCENTILE_HIGH
_VALUE_P_LOW  = REVENUE_PERCENTILE_LOW


# ── Action catalogue ───────────────────────────────────────────────────────────
ACTIONS = {
    "Retain - Premium Offer":   "High-value customer at high churn risk. Escalate to relationship manager. Offer premium loyalty package and dedicated service line.",
    "Retain - Standard Offer":  "Medium-value customer at high churn risk. Assign proactive outreach. Offer fee waiver or product upgrade.",
    "Cross-Sell Opportunity":   "Low churn risk and measurable financial value. Ideal candidate for product expansion (investment, insurance, credit).",
    "Monitor":                  "Elevated risk or declining activity observed. Track engagement metrics over next 30 days before intervention.",
    "No Action":                "Customer appears stable. No immediate risk or opportunity signal detected. Re-evaluate at next scoring cycle.",
}


def _classify_risk(churn_prob: float, p65: float, p35: float) -> str:
    """Percentile-based bucketing so risk levels are always well-populated."""
    if churn_prob >= p65:
        return "High"
    if churn_prob >= p35:
        return "Medium"
    return "Low"


def _classify_value(fv_score: float, high_thr: float, low_thr: float) -> str:
    if fv_score >= high_thr:
        return "High"
    if fv_score >= low_thr:
        return "Medium"
    return "Low"


_DECISION_MATRIX = {
    ("High",   "High"):   "Retain - Premium Offer",
    ("High",   "Medium"): "Retain - Standard Offer",
    ("High",   "Low"):    "Monitor",
    ("Medium", "High"):   "Cross-Sell Opportunity",
    ("Medium", "Medium"): "Monitor",
    ("Medium", "Low"):    "No Action",
    ("Low",    "High"):   "Cross-Sell Opportunity",
    ("Low",    "Medium"): "No Action",
    ("Low",    "Low"):    "No Action",
}


def _priority_score(risk: str, value: str, churn_prob: float, fv_score: float) -> float:
    """Higher = needs attention sooner. Blends urgency (churn) and value (revenue)."""
    risk_weight = {"High": 3, "Medium": 2, "Low": 1}[risk]
    val_weight  = {"High": 3, "Medium": 2, "Low": 1}[value]
    return round(churn_prob * risk_weight * 0.5 + fv_score * val_weight * 0.5, 4)


def apply_recommendations(scoring_table: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    scoring_table : output of clv_scorer.build_scoring_table()

    Returns
    -------
    Full recommendations DataFrame.
    """
    df = scoring_table.copy()

    # Dynamic thresholds — always produce well-populated buckets regardless of model scale
    fv_high   = df["financial_value_score"].quantile(_VALUE_P_HIGH / 100)
    fv_low    = df["financial_value_score"].quantile(_VALUE_P_LOW  / 100)
    churn_p65 = df["churn_probability"].quantile(0.65)
    churn_p35 = df["churn_probability"].quantile(0.35)

    df["risk_level"]    = df["churn_probability"].apply(
        lambda p: _classify_risk(p, churn_p65, churn_p35)
    )
    df["financial_value"] = df["financial_value_score"].apply(
        lambda s: _classify_value(s, fv_high, fv_low)
    )

    df["recommended_action"] = df.apply(
        lambda r: _DECISION_MATRIX[(r["risk_level"], r["financial_value"])], axis=1
    )
    df["action_rationale"] = df["recommended_action"].map(ACTIONS)

    df["priority_score"] = df.apply(
        lambda r: _priority_score(
            r["risk_level"], r["financial_value"],
            r["churn_probability"], r["financial_value_score"]
        ), axis=1
    )

    # Segment-aware override: high-net-worth customers always get premium treatment
    hnw_high_risk = (df["segment"] == "high_net_worth") & (df["risk_level"] == "High")
    df.loc[hnw_high_risk, "recommended_action"] = "Retain - Premium Offer"
    df.loc[hnw_high_risk, "action_rationale"]   = ACTIONS["Retain - Premium Offer"]

    out = PROCESSED_DIR / "recommendations.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info(
        "Recommendations saved → %s\n  Action distribution:\n%s",
        out,
        df["recommended_action"].value_counts().to_string()
    )
    return df
