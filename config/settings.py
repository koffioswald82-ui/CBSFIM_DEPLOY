"""
Central configuration for CBSFIM pipeline.
All tunable parameters live here — no magic numbers in business logic.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
PARQUET_DIR     = DATA_DIR / "parquet"
CUSTOMERS_PQDIR = PARQUET_DIR / "customers"
TRANSACTIONS_PQDIR = PARQUET_DIR / "transactions"

# ── Simulation scale ───────────────────────────────────────────────────────────
N_CUSTOMERS    = 1_000_000
N_TRANSACTIONS = 50_000_000
SIM_MONTHS     = 24
RANDOM_SEED    = 42

# Use LITE mode (pandas) when Spark is unavailable or for quick testing
# Set to False to run full PySpark simulation (requires Java + PySpark)
LITE_MODE = True          # flip to False for full Spark run (1M clients, 50M transactions)
LITE_CUSTOMERS    = 50_000   # DEMO — réduit pour Streamlit Cloud
LITE_TRANSACTIONS = 500_000  # DEMO — réduit pour Streamlit Cloud

# ── Customer segments ──────────────────────────────────────────────────────────
SEGMENTS = {
    "retail":        {"weight": 0.55, "income_mu": 35_000,  "income_sigma": 10_000,  "balance_mu": 5_000,   "balance_sigma": 3_000},
    "premium":       {"weight": 0.30, "income_mu": 85_000,  "income_sigma": 20_000,  "balance_mu": 30_000,  "balance_sigma": 12_000},
    "high_net_worth":{"weight": 0.15, "income_mu": 250_000, "income_sigma": 80_000,  "balance_mu": 200_000, "balance_sigma": 60_000},
}

# ── Transaction parameters ─────────────────────────────────────────────────────
TX_TYPES    = ["payment", "transfer", "withdrawal"]
TX_CHANNELS = ["online", "ATM", "branch"]

TX_AMOUNT_BY_SEGMENT = {
    "retail":         {"mu": 150,   "sigma": 80},
    "premium":        {"mu": 600,   "sigma": 250},
    "high_net_worth": {"mu": 3_000, "sigma": 1_200},
}

# Monthly transaction frequency (avg per customer)
TX_FREQ_BY_SEGMENT = {
    "retail":         15,
    "premium":        25,
    "high_net_worth": 35,
}

# ── ML thresholds ──────────────────────────────────────────────────────────────
CHURN_HIGH_THRESHOLD   = 0.65
CHURN_MEDIUM_THRESHOLD = 0.35

REVENUE_PERCENTILE_HIGH = 75   # top quartile → high financial value
REVENUE_PERCENTILE_LOW  = 25

# ── Business metrics ───────────────────────────────────────────────────────────
CLV_DISCOUNT_RATE   = 0.10   # annual
CLV_YEARS           = 5
REVENUE_MARGIN_RATE = 0.18   # net margin on revenue

# ── BI / Tableau output directories ───────────────────────────────────────────
BI_DIR      = DATA_DIR / "bi"
TABLEAU_DIR = BI_DIR / "tableau"
KPI_DIR     = BI_DIR / "kpis"
SPEC_DIR    = BI_DIR / "specs"

# ── Spark ──────────────────────────────────────────────────────────────────────
SPARK_APP_NAME    = "CBSFIM"
SPARK_MASTER      = "local[*]"
SPARK_DRIVER_MEM  = "4g"
SPARK_EXECUTOR_MEM = "4g"
