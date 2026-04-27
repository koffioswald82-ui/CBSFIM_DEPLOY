"""
Churn Prediction Model
----------------------
XGBoost binary classifier trained on customer + aggregated transaction features.
Output: churn_probability ∈ [0, 1] per customer.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR, CHURN_HIGH_THRESHOLD, CHURN_MEDIUM_THRESHOLD

logger = logging.getLogger(__name__)

MODEL_PATH = PROCESSED_DIR / "churn_model.joblib"
FEATURES   = [
    "age", "income", "account_balance", "tenure_months",
    "tx_count", "tx_avg_amount", "tx_std_amount",
    "tx_recency_days", "online_ratio", "atm_ratio",
    "segment_retail", "segment_premium", "segment_hnw",
]


def _encode_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segment_retail"]  = (df["segment"] == "retail").astype(int)
    df["segment_premium"] = (df["segment"] == "premium").astype(int)
    df["segment_hnw"]     = (df["segment"] == "high_net_worth").astype(int)
    return df


def _build_features(customers: pd.DataFrame, tx_agg: pd.DataFrame) -> pd.DataFrame:
    df = customers.merge(tx_agg, on="customer_id", how="left")
    df["tx_count"]        = df.get("tx_count",        pd.Series(0, index=df.index)).fillna(0)
    df["tx_avg_amount"]   = df.get("tx_avg_amount",   pd.Series(0, index=df.index)).fillna(0)
    df["tx_std_amount"]   = df.get("tx_std_amount",   pd.Series(0, index=df.index)).fillna(0)
    df["tx_recency_days"] = df.get("tx_recency_days", pd.Series(365, index=df.index)).fillna(365)
    df["online_ratio"]    = df.get("online_ratio",    pd.Series(0.5, index=df.index)).fillna(0.5)
    df["atm_ratio"]       = df.get("atm_ratio",       pd.Series(0.3, index=df.index)).fillna(0.3)
    df = _encode_segment(df)
    return df


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer transaction features from raw tx data."""
    from datetime import datetime

    ref_date = transactions["timestamp"].max()
    if hasattr(ref_date, "to_pydatetime"):
        ref_date = ref_date.to_pydatetime()

    agg = transactions.groupby("customer_id").agg(
        tx_count        = ("transaction_id", "count"),
        tx_avg_amount   = ("amount", "mean"),
        tx_std_amount   = ("amount", "std"),
        last_tx_date    = ("timestamp", "max"),
    ).reset_index()

    agg["tx_recency_days"] = (ref_date - pd.to_datetime(agg["last_tx_date"])).dt.days

    channel_counts = (
        transactions.groupby(["customer_id", "channel"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"online": "online_cnt", "ATM": "atm_cnt", "branch": "branch_cnt"})
        .reset_index()
    )
    for col in ["online_cnt", "atm_cnt", "branch_cnt"]:
        if col not in channel_counts.columns:
            channel_counts[col] = 0

    channel_counts["total_cnt"] = (
        channel_counts["online_cnt"] + channel_counts["atm_cnt"] + channel_counts["branch_cnt"]
    )
    channel_counts["online_ratio"] = channel_counts["online_cnt"] / channel_counts["total_cnt"].replace(0, 1)
    channel_counts["atm_ratio"]    = channel_counts["atm_cnt"]    / channel_counts["total_cnt"].replace(0, 1)

    agg = agg.merge(channel_counts[["customer_id", "online_ratio", "atm_ratio"]], on="customer_id", how="left")
    agg["tx_std_amount"] = agg["tx_std_amount"].fillna(0)
    return agg


def train_churn_model(customers: pd.DataFrame, transactions: pd.DataFrame) -> "XGBClassifier":
    """Train XGBoost churn classifier using simulated churn_risk_score as label."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    tx_agg = aggregate_transactions(transactions)
    df     = _build_features(customers, tx_agg)

    # Use simulated churn_risk_score to derive binary label
    y = (df["churn_risk_score"] > CHURN_HIGH_THRESHOLD).astype(int)
    X = df[FEATURES].fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        use_label_encoder=False,
        eval_metric     = "auc",
        random_state    = 42,
        n_jobs          = -1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info("Churn model AUC: %.4f", auc)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Churn model saved → %s", MODEL_PATH)
    return model


def predict_churn(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with customer_id + churn_probability."""
    tx_agg = aggregate_transactions(transactions)
    df     = _build_features(customers, tx_agg)
    X      = df[FEATURES].fillna(0).astype(float)

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded churn model from %s", MODEL_PATH)
    else:
        logger.warning("No saved churn model found — training now...")
        model = train_churn_model(customers, transactions)

    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "customer_id":       df["customer_id"].values,
        "churn_probability": np.round(proba, 4),
    })
