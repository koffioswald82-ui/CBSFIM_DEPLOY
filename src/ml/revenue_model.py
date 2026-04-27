"""
Revenue Prediction Model
------------------------
XGBoost regressor estimating expected annual revenue per customer.
Revenue proxy = annual_tx_volume * margin_rate.
Output: predicted_revenue_annual ∈ ℝ+ per customer.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR, REVENUE_MARGIN_RATE

logger = logging.getLogger(__name__)

MODEL_PATH = PROCESSED_DIR / "revenue_model.joblib"

FEATURES = [
    "age", "income", "account_balance", "tenure_months",
    "tx_count", "tx_avg_amount", "tx_total_volume",
    "online_ratio", "segment_retail", "segment_premium", "segment_hnw",
]


def _build_revenue_target(tx_agg: pd.DataFrame) -> pd.DataFrame:
    """Derive annual revenue from transaction volume."""
    df = tx_agg.copy()
    months_observed  = 24
    months_in_year   = 12
    scale            = months_in_year / months_observed
    df["annual_tx_volume"]       = df["tx_total_volume"] * scale
    df["predicted_revenue_actual"] = (df["annual_tx_volume"] * REVENUE_MARGIN_RATE).round(2)
    return df


def _build_features(customers: pd.DataFrame, tx_agg: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(tx_agg, on="customer_id", how="left")
    for col in ["tx_count", "tx_avg_amount", "tx_total_volume", "online_ratio"]:
        df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0)
    df["segment_retail"]  = (df["segment"] == "retail").astype(int)
    df["segment_premium"] = (df["segment"] == "premium").astype(int)
    df["segment_hnw"]     = (df["segment"] == "high_net_worth").astype(int)
    return df


def aggregate_revenue_features(transactions: pd.DataFrame) -> pd.DataFrame:
    agg = transactions.groupby("customer_id").agg(
        tx_count        = ("transaction_id", "count"),
        tx_avg_amount   = ("amount", "mean"),
        tx_total_volume = ("amount", "sum"),
    ).reset_index()

    channel_counts = (
        transactions.groupby(["customer_id", "channel"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["online", "ATM", "branch"]:
        if col not in channel_counts.columns:
            channel_counts[col] = 0
    channel_counts["total_cnt"]    = channel_counts["online"] + channel_counts["ATM"] + channel_counts["branch"]
    channel_counts["online_ratio"] = channel_counts["online"] / channel_counts["total_cnt"].replace(0, 1)

    agg = agg.merge(channel_counts[["customer_id", "online_ratio"]], on="customer_id", how="left")
    agg["online_ratio"] = agg["online_ratio"].fillna(0.5)
    return agg


def train_revenue_model(customers: pd.DataFrame, transactions: pd.DataFrame):
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    tx_agg = aggregate_revenue_features(transactions)
    tx_agg = _build_revenue_target(tx_agg)
    df     = _build_features(customers, tx_agg)

    y = df["predicted_revenue_actual"].fillna(0).values
    X = df[FEATURES].fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = 42,
        n_jobs          = -1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    logger.info("Revenue model  R²: %.4f  MAE: %.2f", r2_score(y_test, preds), mean_absolute_error(y_test, preds))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Revenue model saved → %s", MODEL_PATH)
    return model


def predict_revenue(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    tx_agg = aggregate_revenue_features(transactions)
    tx_agg = _build_revenue_target(tx_agg)
    df     = _build_features(customers, tx_agg)
    X      = df[FEATURES].fillna(0).astype(float)

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        logger.warning("No saved revenue model — training now...")
        model = train_revenue_model(customers, transactions)

    preds = model.predict(X)
    return pd.DataFrame({
        "customer_id":              df["customer_id"].values,
        "predicted_revenue_annual": np.round(np.clip(preds, 0, None), 2),
    })
