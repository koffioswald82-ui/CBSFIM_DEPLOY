"""
BI Pipeline Orchestrator
-------------------------
Runs the full Business Intelligence layer on top of the existing ML outputs:

  Step 1 — Load:     reads recommendations.parquet + transactions parquet files
  Step 2 — Engineer: builds 40+ features via src/bi/feature_engineering.py
  Step 3 — Build:    creates the three Tableau-ready datasets
  Step 4 — KPIs:     computes the executive KPI report
  Step 5 — Export:   writes CSV + Parquet to data/bi/tableau/
  Step 6 — Spec:     generates the Tableau dashboard blueprint JSON

Prerequisites:
  python pipelines/run_pipeline.py   # must run first to produce recommendations.parquet

Usage:
  python pipelines/run_bi_pipeline.py
"""

import json
import logging
import sys
import time
import os
from pathlib import Path

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import PROCESSED_DIR, TRANSACTIONS_PQDIR, DATA_DIR
from src.bi.feature_engineering import run_feature_engineering
from src.bi.tableau_datasets import (
    build_customer_table,
    build_financial_aggregates,
    build_time_series,
    export_all,
)
from src.bi.kpi_engine import compute_all_kpis
from src.bi.dashboard_spec import build_dashboard_spec

# ── Output directories ────────────────────────────────────────────────────────
BI_DIR      = DATA_DIR / "bi"
TABLEAU_DIR = BI_DIR / "tableau"
KPI_DIR     = BI_DIR / "kpis"
SPEC_DIR    = BI_DIR / "specs"


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bi_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str, width: int = 62) -> None:
    logger.info("=" * width)
    logger.info("  %s", text)
    logger.info("=" * width)


def _step(n: int, total: int, label: str) -> None:
    logger.info("")
    logger.info("[STEP %d/%d]  %s", n, total, label)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads recommendations + all transaction parquet shards."""
    rec_path = PROCESSED_DIR / "recommendations.parquet"
    if not rec_path.exists():
        raise FileNotFoundError(
            f"\n  recommendations.parquet not found at:\n  {rec_path}\n\n"
            "  Run the main pipeline first:\n"
            "    python pipelines/run_pipeline.py\n"
        )

    logger.info("  Loading recommendations.parquet ...")
    recs = pd.read_parquet(rec_path)
    logger.info("  → %d customers, %d columns", len(recs), len(recs.columns))

    logger.info("  Loading transactions (all parquet shards) ...")
    tx = pd.read_parquet(TRANSACTIONS_PQDIR)
    logger.info("  → %d transactions, %d columns", len(tx), len(tx.columns))

    return recs, tx


def _numpy_to_python(obj):
    """Convert numpy scalars to native Python types so json.dump never falls back to str()."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_numpy_to_python)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_bi_pipeline() -> dict:
    t0 = time.time()
    N_STEPS = 6

    _banner("CBSFIM  BI PIPELINE  —  Tableau Dataset Builder")

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    _step(1, N_STEPS, "Load ML outputs")
    recommendations, transactions = _load_inputs()

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    _step(2, N_STEPS, "Advanced Feature Engineering")
    enriched = run_feature_engineering(recommendations, transactions)

    # Save the full enriched master table for ad-hoc analysis
    BI_DIR.mkdir(parents=True, exist_ok=True)
    enriched_path = BI_DIR / "enriched_customers.parquet"
    enriched.to_parquet(enriched_path, index=False)
    logger.info(
        "  Enriched master saved → %s  (%d rows × %d cols)",
        enriched_path, len(enriched), len(enriched.columns),
    )

    # ── Step 3: Build Tableau datasets ────────────────────────────────────────
    _step(3, N_STEPS, "Build Tableau Datasets (A / B / C)")

    logger.info("  A. Customer Table ...")
    customer_df   = build_customer_table(enriched)

    logger.info("  B. Financial Aggregates ...")
    financial_df  = build_financial_aggregates(enriched)

    logger.info("  C. Time Series ...")
    timeseries_df = build_time_series(transactions, enriched)

    # ── Step 4: KPI computation ───────────────────────────────────────────────
    _step(4, N_STEPS, "Compute Executive KPI Report")
    kpi_report = compute_all_kpis(enriched, timeseries_df)

    kpi_path = KPI_DIR / "kpi_report.json"
    _save_json(kpi_report, kpi_path)
    logger.info("  KPI report saved → %s", kpi_path)

    # ── Step 5: Export datasets ───────────────────────────────────────────────
    _step(5, N_STEPS, "Export CSV + Parquet Datasets")
    file_paths = export_all(customer_df, financial_df, timeseries_df, TABLEAU_DIR)

    # ── Step 6: Dashboard specification ──────────────────────────────────────
    _step(6, N_STEPS, "Generate Tableau Dashboard Blueprint")
    dashboard_spec = build_dashboard_spec(kpi_report["summary"])

    spec_path = SPEC_DIR / "dashboard_spec.json"
    _save_json(dashboard_spec, spec_path)
    logger.info("  Dashboard spec saved → %s", spec_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    kpis    = kpi_report["kpis"]
    summ    = kpi_report["summary"]

    _banner(f"BI PIPELINE COMPLETE  ({elapsed:.1f}s)")
    logger.info("  Customers processed        : %d", len(enriched))
    logger.info("  Total Annual Revenue       : $%s",
                f"{summ['total_revenue']:,.0f}")
    logger.info("  Revenue at Risk            : $%s  (%.1f%%)",
                f"{summ['total_revenue_at_risk']:,.0f}", summ["rar_pct"])
    logger.info("  Portfolio Churn Rate       : %.1f%%", summ["churn_rate_pct"])
    logger.info("  Total 5-yr CLV             : $%s",
                f"{summ['total_clv']:,.0f}")
    logger.info("  Net Portfolio Profit       : $%s",
                f"{summ['total_net_profit']:,.0f}")
    logger.info("  Critical Accounts (HVAR)   : %d", summ["critical_customers"])
    logger.info("")
    logger.info("  OUTPUT FILES")
    logger.info("  %-40s %s", "enriched_customers.parquet", enriched_path)
    for name, path in file_paths.items():
        logger.info("  %-40s %s", name, path)
    logger.info("  %-40s %s", "kpi_report.json", kpi_path)
    logger.info("  %-40s %s", "dashboard_spec.json", spec_path)
    logger.info("=" * 62)

    return {
        "enriched":            enriched,
        "customer_table":      customer_df,
        "financial_aggregates":financial_df,
        "time_series":         timeseries_df,
        "kpi_report":          kpi_report,
        "dashboard_spec":      dashboard_spec,
        "file_paths":          file_paths,
    }


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_bi_pipeline()
