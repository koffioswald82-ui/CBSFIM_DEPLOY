"""
CBSFIM Master Pipeline
======================
Orchestrates the full end-to-end pipeline:
  1. Customer data simulation
  2. Transaction data simulation
  3. Churn model training + prediction
  4. Revenue model training + prediction
  5. CLV scoring
  6. AI recommendation engine
  7. Business metrics computation

Usage
-----
    # Lite mode (default, fast — pandas, ~200k customers, ~2M transactions)
    python pipelines/run_pipeline.py

    # Full Spark mode (1M customers, 50M transactions)
    python pipelines/run_pipeline.py --full

    # Skip training, re-use saved models
    python pipelines/run_pipeline.py --no-train
"""

import sys
import os
import time
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CBSFIM.pipeline")


def _step(name: str):
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)


def run(full_mode: bool = False, skip_train: bool = False):
    import config.settings as cfg

    if full_mode:
        cfg.LITE_MODE = False
        logger.info("Running FULL PySpark mode — %d customers, %d transactions",
                    cfg.N_CUSTOMERS, cfg.N_TRANSACTIONS)
    else:
        cfg.LITE_MODE = True
        logger.info("Running LITE mode — %d customers, %d transactions",
                    cfg.LITE_CUSTOMERS, cfg.LITE_TRANSACTIONS)

    spark = None
    if not cfg.LITE_MODE:
        from src.simulation.spark_session import get_spark
        spark = get_spark()

    t0 = time.time()

    # ── Step 1: Simulate customers ─────────────────────────────────────────────
    _step("1 / 7  Customer Simulation")
    from src.simulation.customer_generator import generate_customers
    customers = generate_customers(spark)
    logger.info("  → %d customers generated", len(customers))

    # ── Step 2: Simulate transactions ──────────────────────────────────────────
    _step("2 / 7  Transaction Simulation")
    from src.simulation.transaction_generator import generate_transactions
    transactions = generate_transactions(customers, spark)
    logger.info("  → %d transactions generated", len(transactions))

    # ── Step 3: Churn model ────────────────────────────────────────────────────
    _step("3 / 7  Churn Model")
    from src.ml.churn_model import train_churn_model, predict_churn
    if not skip_train:
        train_churn_model(customers, transactions)
    churn_preds = predict_churn(customers, transactions)
    logger.info("  → Churn predictions: %d rows  |  avg prob: %.4f",
                len(churn_preds), churn_preds["churn_probability"].mean())

    # ── Step 4: Revenue model ──────────────────────────────────────────────────
    _step("4 / 7  Revenue Model")
    from src.ml.revenue_model import train_revenue_model, predict_revenue
    if not skip_train:
        train_revenue_model(customers, transactions)
    revenue_preds = predict_revenue(customers, transactions)
    logger.info("  → Revenue predictions: %d rows  |  avg: $%.2f/yr",
                len(revenue_preds), revenue_preds["predicted_revenue_annual"].mean())

    # ── Step 5: CLV scoring ────────────────────────────────────────────────────
    _step("5 / 7  CLV Scoring")
    from src.ml.clv_scorer import build_scoring_table
    scoring = build_scoring_table(customers, churn_preds, revenue_preds)
    logger.info("  → Scoring table: %d rows  |  avg CLV: $%.2f",
                len(scoring), scoring["clv"].mean())

    # ── Step 6: AI recommendations ─────────────────────────────────────────────
    _step("6 / 7  AI Recommendation Engine")
    from src.engine.recommendation_engine import apply_recommendations
    recommendations = apply_recommendations(scoring)
    logger.info("  → Recommendations:\n%s",
                recommendations["recommended_action"].value_counts().to_string())

    # ── Step 7: Business metrics ───────────────────────────────────────────────
    _step("7 / 7  Business Metrics")
    from src.metrics.business_metrics import compute_metrics
    metrics = compute_metrics(recommendations)

    elapsed = time.time() - t0

    logger.info("")
    logger.info("━" * 60)
    logger.info("PIPELINE COMPLETE  —  %.1f seconds", elapsed)
    logger.info("━" * 60)
    logger.info("  Customers analysed   : %s",  f"{metrics['n_customers']:,}")
    logger.info("  Churn rate (High)    : %.2f%%", metrics["churn_rate_pct"])
    logger.info("  Revenue at Risk      : $%s",  f"{metrics['revenue_at_risk']:,.0f}")
    logger.info("  Expected Loss        : $%s",  f"{metrics['total_expected_loss']:,.0f}")
    logger.info("  High-value at Risk   : %d",   metrics["high_value_at_risk"])
    logger.info("  Total CLV            : $%s",  f"{metrics['total_clv']:,.0f}")
    logger.info("")
    logger.info("  Outputs in:  %s", cfg.PROCESSED_DIR)
    logger.info("  Dashboard:   streamlit run dashboard/app.py")
    logger.info("━" * 60)

    if spark:
        from src.simulation.spark_session import stop_spark
        stop_spark(spark)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBSFIM End-to-End Pipeline")
    parser.add_argument("--full",     action="store_true", help="Run full PySpark mode (1M customers, 50M tx)")
    parser.add_argument("--no-train", action="store_true", help="Skip model training, reuse saved models")
    args = parser.parse_args()
    run(full_mode=args.full, skip_train=args.no_train)
