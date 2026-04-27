"""
Advanced Feature Engineering for the BI Layer
-----------------------------------------------
Transforms raw transactions + ML recommendations into a 40+ column enriched
customer dataset ready for Tableau consumption and executive KPI computation.

Feature groups built here:
  BEHAVIOR    — transaction frequency windows, trend slope, volatility
  CHURN SIG   — recency, inactivity streak, balance decay proxy
  FINANCIAL   — margin, cost-to-serve, net profit, revenue at risk
  COMPOSITE   — engagement index, risk-adjusted value, KMeans clusters
"""

import logging
import sys
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import REVENUE_MARGIN_RATE

logger = logging.getLogger(__name__)

# ── Simulation calendar boundaries ────────────────────────────────────────────
SIM_START = pd.Timestamp("2023-01-01")
SIM_END   = pd.Timestamp("2024-12-24")
TOTAL_OBS_DAYS = (SIM_END - SIM_START).days          # 723

DATE_30D_AGO = SIM_END - pd.Timedelta(days=30)       # 2024-11-24
DATE_90D_AGO = SIM_END - pd.Timedelta(days=90)       # 2024-09-25

# Realistic annual cost-to-serve benchmarks by banking segment (USD)
COST_TO_SERVE = {
    "retail":         180.0,   # high-volume, low-touch, mostly digital
    "premium":        320.0,   # relationship manager, some branch visits
    "high_net_worth": 600.0,   # dedicated advisor, premium services
}

N_MONTHS = 24   # simulation duration in months


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRANSACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-customer behavioral features from the full 24-month transaction
    history.  Returns a DataFrame indexed by customer_id with 9 feature columns.
    Performance note: uses vectorised monthly pivot instead of per-row apply.
    """
    logger.info("  Building transaction features from %d rows...", len(transactions))

    tx = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(tx["timestamp"]):
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    # ── Time-windowed counts (30d / 90d before end of simulation) ─────────────
    count_30d = (
        tx[tx["timestamp"] >= DATE_30D_AGO]
        .groupby("customer_id")["transaction_id"].count()
        .rename("transaction_count_30d")
    )
    count_90d = (
        tx[tx["timestamp"] >= DATE_90D_AGO]
        .groupby("customer_id")["transaction_id"].count()
        .rename("transaction_count_90d")
    )

    # ── Recency ────────────────────────────────────────────────────────────────
    days_since = (
        (SIM_END - tx.groupby("customer_id")["timestamp"].max())
        .dt.days
        .rename("days_since_last_transaction")
    )

    # ── Active days ratio ──────────────────────────────────────────────────────
    # Unique calendar days a customer made at least one transaction / 723 total days
    tx["_date"] = tx["timestamp"].dt.normalize()
    unique_days = (
        tx.drop_duplicates(["customer_id", "_date"])
        .groupby("customer_id")["_date"]
        .count()
    )
    active_days_ratio = (unique_days / TOTAL_OBS_DAYS).clip(0, 1).rename("active_days_ratio")

    # ── Average transaction amount ─────────────────────────────────────────────
    avg_amount = tx.groupby("customer_id")["amount"].mean().rename("avg_transaction_amount")

    # ── Per-customer monthly spending pivot (shape: n_customers × 24) ─────────
    # The "month" column (0-23) maps directly to simulation months
    monthly_pivot = (
        tx.groupby(["customer_id", "month"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=list(range(N_MONTHS)), fill_value=0.0)
    )

    # ── Spending volatility: std across the 24 monthly buckets ────────────────
    spending_vol = monthly_pivot.std(axis=1).rename("spending_volatility")

    # ── Spending trend slope: vectorised OLS over 24 equally-spaced months ────
    x = np.arange(N_MONTHS, dtype=float)
    x_c = x - x.mean()                        # centred x
    x_ss = float((x_c ** 2).sum())            # sum of squares of x
    spending_slope = pd.Series(
        monthly_pivot.values.dot(x_c) / x_ss,
        index=monthly_pivot.index,
        name="spending_trend_slope",
    )

    # ── Inactivity streak: consecutive months with zero transactions at tail ──
    active_mask = (monthly_pivot.values > 0)   # bool matrix n_customers × 24

    def _trailing_zeros(row: np.ndarray) -> int:
        """Count consecutive inactive months ending at the most recent month."""
        streak = 0
        for v in row[::-1]:
            if not v:
                streak += 1
            else:
                break
        return streak

    inact_streak = pd.Series(
        [_trailing_zeros(r) for r in active_mask],
        index=monthly_pivot.index,
        name="inactivity_streak",
    )

    # ── Online channel ratio ───────────────────────────────────────────────────
    channel_pivot = (
        tx.groupby(["customer_id", "channel"])["transaction_id"]
        .count()
        .unstack(fill_value=0)
    )
    total_tx_counts = channel_pivot.sum(axis=1).replace(0, np.nan)
    if "online" in channel_pivot.columns:
        online_ratio = (channel_pivot["online"] / total_tx_counts).fillna(0)
    else:
        online_ratio = pd.Series(0.0, index=channel_pivot.index)
    online_ratio = online_ratio.rename("online_channel_ratio")

    # ── Combine ────────────────────────────────────────────────────────────────
    result = pd.concat(
        [count_30d, count_90d, days_since, active_days_ratio,
         avg_amount, spending_slope, spending_vol, inact_streak, online_ratio],
        axis=1,
    ).fillna(0)

    logger.info(
        "  Transaction features complete: %d customers × %d features",
        len(result), len(result.columns),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. FINANCIAL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds margin, cost-to-serve, net profit, and revenue-at-risk columns.
    Requires: segment, predicted_revenue_annual, churn_probability, account_balance.
    """
    out = df.copy()

    # Cost to serve — segment driven (annual, USD)
    out["cost_to_serve"] = out["segment"].map(COST_TO_SERVE).fillna(250.0)

    # Revenue and margin
    out["total_revenue_generated"]    = out["predicted_revenue_annual"].round(2)
    out["estimated_margin_per_client"]= (out["predicted_revenue_annual"] * REVENUE_MARGIN_RATE).round(2)

    # Net profit = margin earned minus the cost to serve that customer
    out["net_profit_per_client"] = (
        out["estimated_margin_per_client"] - out["cost_to_serve"]
    ).round(2)

    # Revenue at risk: probability-weighted annual revenue loss
    out["revenue_at_risk"] = (
        out["predicted_revenue_annual"] * out["churn_probability"]
    ).round(2)

    # Balance decay rate: proxy using churn_probability × the 60% activity
    # decay baked into the simulation for high-risk customers
    out["balance_decay_rate"] = (out["churn_probability"] * 0.60).round(4)

    # Human-readable tenure
    out["customer_tenure_years"] = (out["tenure_months"] / 12).round(2)

    # Alias financial_value_score as customer_value_score for schema clarity
    out["customer_value_score"] = out["financial_value_score"].round(4)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE FEATURES & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def build_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds three composite indicators and two cluster label columns.

    engagement_index         [0,1]  — weighted behavioral activity score
    risk_adjusted_value_score[0,1]  — CLV score discounted by churn probability
    churn_probability_pct    [0,100]— display-friendly churn percentage

    risk_cluster             low / medium / high  — KMeans on churn+profit+engagement
    value_cluster            low / medium / high  — CLV percentile bands
    """
    out = df.copy()

    # ── Engagement Index ───────────────────────────────────────────────────────
    # Recency score: 1 = transacted yesterday, 0 = no transaction in full period
    max_recency = max(float(out["days_since_last_transaction"].max()), 1.0)
    recency_score = 1.0 - (out["days_since_last_transaction"] / max_recency).clip(0, 1)

    max_tx90 = max(float(out["transaction_count_90d"].max()), 1.0)
    tx_freq_norm = (out["transaction_count_90d"] / max_tx90).clip(0, 1)

    online_col = out["online_channel_ratio"] if "online_channel_ratio" in out.columns \
        else pd.Series(0.0, index=out.index)

    out["engagement_index"] = (
        0.30 * out["active_days_ratio"]
        + 0.30 * tx_freq_norm
        + 0.20 * online_col
        + 0.20 * recency_score
    ).round(4).clip(0, 1)

    # ── Risk-Adjusted Value Score ──────────────────────────────────────────────
    # Customers with high CLV but high churn get a heavily discounted score
    out["risk_adjusted_value_score"] = (
        out["financial_value_score"] * (1.0 - out["churn_probability"])
    ).round(4)

    out["churn_probability_pct"] = (out["churn_probability"] * 100).round(1)

    # ── Risk Cluster (KMeans on 3 financial-behavioral dimensions) ────────────
    cluster_cols = ["churn_probability", "net_profit_per_client", "engagement_index"]
    available    = [c for c in cluster_cols if c in out.columns]

    try:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(out[available].fillna(0))
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Assign semantic names: sort clusters by mean churn probability
        cluster_churn = (
            pd.Series(out["churn_probability"].values)
            .groupby(labels).mean()
            .sort_values()
        )
        risk_map = {
            cluster_churn.index[0]: "low",
            cluster_churn.index[1]: "medium",
            cluster_churn.index[2]: "high",
        }
        out["risk_cluster"] = pd.Categorical(
            [risk_map[c] for c in labels],
            categories=["low", "medium", "high"],
            ordered=True,
        )
        logger.info("  KMeans risk clustering done.")
    except Exception as exc:
        logger.warning("  KMeans failed (%s) — using percentile fallback.", exc)
        p33 = out["churn_probability"].quantile(0.33)
        p66 = out["churn_probability"].quantile(0.66)
        out["risk_cluster"] = pd.cut(
            out["churn_probability"],
            bins=[-np.inf, p33, p66, np.inf],
            labels=["low", "medium", "high"],
        )

    # ── Value Cluster (CLV percentile bands) ──────────────────────────────────
    p33_clv = out["clv"].quantile(0.33)
    p66_clv = out["clv"].quantile(0.66)
    out["value_cluster"] = pd.cut(
        out["clv"],
        bins=[-np.inf, p33_clv, p66_clv, np.inf],
        labels=["low", "medium", "high"],
    )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(
    recommendations: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.  Returns a single enriched customer
    DataFrame (one row per customer, 40+ columns) ready for Tableau export.
    """
    logger.info("=== Feature Engineering Pipeline ===")

    tx_features = build_transaction_features(transactions)

    enriched = recommendations.join(tx_features, on="customer_id", how="left")

    # Fill zero for customers who appear in recommendations but had no transactions
    # captured in the window (edge case: very new or completely inactive customers)
    for col in tx_features.columns:
        enriched[col] = enriched[col].fillna(0)

    enriched = build_financial_features(enriched)
    enriched = build_composite_features(enriched)

    logger.info(
        "Feature engineering complete: %d rows × %d columns",
        len(enriched), len(enriched.columns),
    )
    return enriched
