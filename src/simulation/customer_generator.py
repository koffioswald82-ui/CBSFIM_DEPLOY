"""
Customer dataset generator.

Full mode  → PySpark DataFrame, written to partitioned Parquet.
Lite mode  → pandas DataFrame (fast local dev / no Spark required).

Schema
------
customer_id        : string
age                : int
income             : float
account_balance    : float
segment            : string  (retail | premium | high_net_worth)
churn_risk_score   : float   [0, 1]
tenure_months      : int
region             : string
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    SEGMENTS, RANDOM_SEED, LITE_CUSTOMERS, N_CUSTOMERS,
    LITE_MODE, CUSTOMERS_PQDIR,
)

logger = logging.getLogger(__name__)

REGIONS = ["North", "South", "East", "West", "Central"]


def _build_customers_pandas(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    seg_names   = list(SEGMENTS.keys())
    seg_weights = [SEGMENTS[s]["weight"] for s in seg_names]
    segments    = rng.choice(seg_names, size=n, p=seg_weights)

    ages    = rng.integers(18, 75, size=n)
    regions = rng.choice(REGIONS, size=n)

    incomes   = np.empty(n)
    balances  = np.empty(n)
    churn_risk = np.empty(n)
    tenure    = rng.integers(1, 180, size=n)   # months

    for i, seg in enumerate(segments):
        cfg = SEGMENTS[seg]
        incomes[i]  = max(0, rng.normal(cfg["income_mu"],  cfg["income_sigma"]))
        balances[i] = max(0, rng.normal(cfg["balance_mu"], cfg["balance_sigma"]))

        # Business rule: higher income → lower base churn; retail → higher churn
        if seg == "retail":
            base_churn = rng.beta(2, 3)      # skewed toward 0.4
        elif seg == "premium":
            base_churn = rng.beta(1.5, 4)   # lower
        else:
            base_churn = rng.beta(1, 5)      # lowest

        # Income penalty: low income increases churn
        income_penalty = max(0, (50_000 - incomes[i]) / 200_000)
        churn_risk[i]  = float(np.clip(base_churn + income_penalty * 0.2, 0.0, 1.0))

    df = pd.DataFrame({
        "customer_id":       [f"CID_{i:08d}" for i in range(n)],
        "age":               ages.astype(int),
        "income":            np.round(incomes, 2),
        "account_balance":   np.round(balances, 2),
        "segment":           segments,
        "churn_risk_score":  np.round(churn_risk, 4),
        "tenure_months":     tenure.astype(int),
        "region":            regions,
    })
    return df


def generate_customers_lite() -> pd.DataFrame:
    """Generate customers with pandas (lite / dev mode)."""
    n = LITE_CUSTOMERS
    logger.info("Generating %d customers (pandas lite mode)...", n)
    df = _build_customers_pandas(n)
    out = CUSTOMERS_PQDIR / "customers.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Customers saved → %s", out)
    return df


def generate_customers_spark(spark) -> "pyspark.sql.DataFrame":
    """Generate 1M customers using PySpark (full production mode)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, FloatType
    )

    logger.info("Generating %d customers (PySpark full mode)...", N_CUSTOMERS)

    # Generate with pandas in chunks, then parallelize
    chunk_size = 200_000
    chunks = []
    seed = RANDOM_SEED
    total = 0
    while total < N_CUSTOMERS:
        n = min(chunk_size, N_CUSTOMERS - total)
        chunk = _build_customers_pandas(n, seed=seed + total)
        chunks.append(chunk)
        total += n

    pdf = pd.concat(chunks, ignore_index=True)
    pdf["customer_id"] = [f"CID_{i:08d}" for i in range(len(pdf))]

    schema = StructType([
        StructField("customer_id",      StringType(),  False),
        StructField("age",              IntegerType(), False),
        StructField("income",           FloatType(),   False),
        StructField("account_balance",  FloatType(),   False),
        StructField("segment",          StringType(),  False),
        StructField("churn_risk_score", FloatType(),   False),
        StructField("tenure_months",    IntegerType(), False),
        StructField("region",           StringType(),  False),
    ])

    # Cast to expected types before creating Spark DF
    pdf = pdf.astype({
        "age": "int32", "income": "float32", "account_balance": "float32",
        "churn_risk_score": "float32", "tenure_months": "int32"
    })

    sdf = spark.createDataFrame(pdf, schema=schema)
    sdf.write.mode("overwrite").parquet(str(CUSTOMERS_PQDIR))
    logger.info("Customers written to Parquet → %s  (partitions: %d)", CUSTOMERS_PQDIR, sdf.rdd.getNumPartitions())
    return sdf


def generate_customers(spark=None) -> pd.DataFrame:
    """Entry point — dispatches to lite or full Spark mode."""
    if LITE_MODE or spark is None:
        return generate_customers_lite()
    return generate_customers_spark(spark).toPandas()
