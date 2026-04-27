"""
Transaction dataset generator.

Schema
------
transaction_id  : string
customer_id     : string
timestamp       : datetime64
amount          : float
type            : string  (payment | transfer | withdrawal)
channel         : string  (online | ATM | branch)
month           : int
"""

import numpy as np
import pandas as pd
import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TX_TYPES, TX_CHANNELS, TX_AMOUNT_BY_SEGMENT, TX_FREQ_BY_SEGMENT,
    SIM_MONTHS, RANDOM_SEED, LITE_MODE, LITE_TRANSACTIONS,
    N_TRANSACTIONS, TRANSACTIONS_PQDIR,
)

logger = logging.getLogger(__name__)

SIM_START = datetime(2023, 1, 1)


def _monthly_spend_factor(month_index: int) -> float:
    """Seasonality: peaks in Dec (index 11), dips in Jan/Feb."""
    angle = 2 * np.pi * month_index / 12
    return 1.0 + 0.25 * np.sin(angle - np.pi / 2 + np.pi)  # ~0.75–1.25


def _churn_activity_decay(churn_risk: float, month_index: int) -> float:
    """High-churn customers reduce activity over time."""
    decay = 1.0 - churn_risk * (month_index / SIM_MONTHS) * 0.6
    return max(0.1, decay)


def _generate_chunk(
    customers: pd.DataFrame,
    target_tx: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_customers = len(customers)
    # Distribute transactions proportionally to frequency weights
    freq_weights = customers["segment"].map(TX_FREQ_BY_SEGMENT).values.astype(float)
    freq_weights /= freq_weights.sum()

    # Sample customer indices for each transaction
    cust_indices = rng.choice(n_customers, size=target_tx, p=freq_weights)
    cust_rows    = customers.iloc[cust_indices].reset_index(drop=True)

    # Random month in [0, SIM_MONTHS)
    month_idx = rng.integers(0, SIM_MONTHS, size=target_tx)
    day_in_month = rng.integers(0, 28, size=target_tx)
    hour  = rng.integers(0, 24, size=target_tx)
    minute = rng.integers(0, 60, size=target_tx)

    timestamps = [
        SIM_START + timedelta(days=int(m * 30.44) + int(d), hours=int(h), minutes=int(mi))
        for m, d, h, mi in zip(month_idx, day_in_month, hour, minute)
    ]

    # Amount — segment-dependent + seasonality + churn decay
    amounts = np.empty(target_tx)
    for seg, params in TX_AMOUNT_BY_SEGMENT.items():
        mask = (cust_rows["segment"] == seg).values
        n_seg = mask.sum()
        if n_seg == 0:
            continue
        base = rng.normal(params["mu"], params["sigma"], size=n_seg)
        season = np.array([_monthly_spend_factor(m) for m in month_idx[mask]])
        churn_dec = np.array([
            _churn_activity_decay(cr, mi)
            for cr, mi in zip(cust_rows["churn_risk_score"].values[mask], month_idx[mask])
        ])
        amounts[mask] = np.clip(base * season * churn_dec, 1.0, None)

    # Premium clients: stable amounts (lower variance)
    premium_mask = (cust_rows["segment"] == "premium").values
    amounts[premium_mask] = np.clip(amounts[premium_mask], 50, 5_000)

    tx_types   = rng.choice(TX_TYPES,    size=target_tx, p=[0.50, 0.30, 0.20])
    tx_channels = rng.choice(TX_CHANNELS, size=target_tx, p=[0.60, 0.25, 0.15])

    df = pd.DataFrame({
        "transaction_id": [f"TID_{seed}_{i:010d}" for i in range(target_tx)],
        "customer_id":    cust_rows["customer_id"].values,
        "timestamp":      timestamps,
        "amount":         np.round(amounts, 2),
        "type":           tx_types,
        "channel":        tx_channels,
        "month":          month_idx.astype(int),
        "segment":        cust_rows["segment"].values,
    })
    return df


def generate_transactions_lite(customers: pd.DataFrame) -> pd.DataFrame:
    n = LITE_TRANSACTIONS
    logger.info("Generating %s transactions (pandas lite mode)...", f"{n:,}")

    chunk_size = 1_000_000   # 1M par chunk — écriture immédiate, pas de concat en mémoire
    paths      = []
    seed       = RANDOM_SEED
    generated  = 0
    part_id    = 0

    TRANSACTIONS_PQDIR.mkdir(parents=True, exist_ok=True)

    # Supprime les anciens fichiers pour éviter les doublons
    for old in TRANSACTIONS_PQDIR.glob("part_*.parquet"):
        old.unlink()

    while generated < n:
        batch = min(chunk_size, n - generated)
        chunk = _generate_chunk(customers, batch, seed + generated)

        out_path = TRANSACTIONS_PQDIR / f"part_{part_id:04d}.parquet"
        chunk.to_parquet(out_path, index=False, engine="pyarrow")
        paths.append(out_path)

        generated += batch
        part_id   += 1
        logger.info("  %s / %s transactions écrites", f"{generated:,}", f"{n:,}")
        del chunk  # libère la mémoire immédiatement

    logger.info("Toutes les transactions écrites → %s", TRANSACTIONS_PQDIR)

    # Relit depuis Parquet pour retourner le DataFrame complet
    return pd.read_parquet(TRANSACTIONS_PQDIR, engine="pyarrow")


def generate_transactions_spark(spark, customers_sdf) -> "pyspark.sql.DataFrame":
    """Full PySpark transaction generation — 50M rows in partitioned Parquet."""
    from pyspark.sql import functions as F

    customers_pd = customers_sdf.toPandas()
    logger.info("Generating %d transactions (PySpark full mode)...", N_TRANSACTIONS)

    chunk_size = 1_000_000
    chunks_pd  = []
    generated  = 0
    part_id    = 0

    while generated < N_TRANSACTIONS:
        batch = min(chunk_size, N_TRANSACTIONS - generated)
        chunk = _generate_chunk(customers_pd, batch, seed=RANDOM_SEED + generated)
        chunks_pd.append(chunk)
        generated += batch
        part_id   += 1

        if len(chunks_pd) >= 5:
            _flush_to_spark_parquet(spark, chunks_pd, TRANSACTIONS_PQDIR)
            chunks_pd = []
            logger.info("  Flushed %d / %d", generated, N_TRANSACTIONS)

    if chunks_pd:
        _flush_to_spark_parquet(spark, chunks_pd, TRANSACTIONS_PQDIR)

    sdf = spark.read.parquet(str(TRANSACTIONS_PQDIR))
    logger.info("Transactions done — total rows: %d", sdf.count())
    return sdf


def _flush_to_spark_parquet(spark, chunks: list, out_dir) -> None:
    pdf = pd.concat(chunks, ignore_index=True)
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("append").partitionBy("segment", "month").parquet(str(out_dir))


def generate_transactions(customers: pd.DataFrame, spark=None) -> pd.DataFrame:
    if LITE_MODE or spark is None:
        return generate_transactions_lite(customers)
    cust_sdf = spark.createDataFrame(customers)
    return generate_transactions_spark(spark, cust_sdf).toPandas()
