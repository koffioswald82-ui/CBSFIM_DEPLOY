"""
Customer Value Scoring (CLV)
----------------------------
Combines churn probability + predicted revenue into a composite
Customer Lifetime Value score and a normalized financial_value_score.

CLV formula:
    CLV = annual_revenue * margin_rate * (1 - churn_prob) *
          sum_{t=1}^{T} (1/(1+r))^t

where r = discount rate, T = projection years.
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import CLV_DISCOUNT_RATE, CLV_YEARS, REVENUE_MARGIN_RATE, PROCESSED_DIR

logger = logging.getLogger(__name__)


def _discount_factor(rate: float = CLV_DISCOUNT_RATE, years: int = CLV_YEARS) -> float:
    """Sum of discount factors over projection horizon."""
    return sum(1 / (1 + rate) ** t for t in range(1, years + 1))


DISCOUNT_FACTOR = _discount_factor()


def compute_clv(churn_preds: pd.DataFrame, revenue_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    churn_preds   : DataFrame with [customer_id, churn_probability]
    revenue_preds : DataFrame with [customer_id, predicted_revenue_annual]

    Returns
    -------
    DataFrame with [customer_id, clv, financial_value_score]
    """
    df = churn_preds.merge(revenue_preds, on="customer_id", how="inner")

    survival_prob = 1.0 - df["churn_probability"]
    df["clv"] = (
        df["predicted_revenue_annual"]
        * REVENUE_MARGIN_RATE
        * survival_prob
        * DISCOUNT_FACTOR
    ).round(2)

    # Normalize to [0, 1] financial value score
    clv_min = df["clv"].min()
    clv_max = df["clv"].max()
    if clv_max > clv_min:
        df["financial_value_score"] = ((df["clv"] - clv_min) / (clv_max - clv_min)).round(4)
    else:
        df["financial_value_score"] = 0.5

    logger.info(
        "CLV computed — mean: %.2f  median: %.2f  max: %.2f",
        df["clv"].mean(), df["clv"].median(), df["clv"].max()
    )
    return df[["customer_id", "clv", "financial_value_score"]]


def build_scoring_table(
    customers:    pd.DataFrame,
    churn_preds:  pd.DataFrame,
    revenue_preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Master customer scoring table joining all predictions.

    Returns full DataFrame with:
        customer_id, segment, age, income, account_balance,
        churn_probability, predicted_revenue_annual, clv, financial_value_score
    """
    clv_df = compute_clv(churn_preds, revenue_preds)

    scoring = (
        customers[["customer_id", "segment", "age", "income",
                   "account_balance", "tenure_months", "region", "churn_risk_score"]]
        .merge(churn_preds,  on="customer_id", how="left")
        .merge(revenue_preds, on="customer_id", how="left")
        .merge(clv_df,        on="customer_id", how="left")
    )

    # Fill missing predictions (customers with no transactions)
    scoring["churn_probability"]        = scoring["churn_probability"].fillna(0.5)
    scoring["predicted_revenue_annual"] = scoring["predicted_revenue_annual"].fillna(0)
    scoring["clv"]                      = scoring["clv"].fillna(0)
    scoring["financial_value_score"]    = scoring["financial_value_score"].fillna(0)

    out = PROCESSED_DIR / "scoring_table.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scoring.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Scoring table saved → %s  (%d rows)", out, len(scoring))
    return scoring
