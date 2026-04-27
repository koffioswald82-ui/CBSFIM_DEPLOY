"""
Spark session factory — singleton pattern with graceful fallback logging.
"""

import sys
import logging

logger = logging.getLogger(__name__)


def get_spark():
    """Return a configured SparkSession; exits cleanly if PySpark unavailable."""
    try:
        from pyspark.sql import SparkSession
        from config.settings import SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEM, SPARK_EXECUTOR_MEM

        spark = (
            SparkSession.builder
            .appName(SPARK_APP_NAME)
            .master(SPARK_MASTER)
            .config("spark.driver.memory", SPARK_DRIVER_MEM)
            .config("spark.executor.memory", SPARK_EXECUTOR_MEM)
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession initialized — master: %s", SPARK_MASTER)
        return spark

    except ImportError:
        logger.error("PySpark not installed. Install with: pip install pyspark")
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to start Spark: %s", exc)
        sys.exit(1)


def stop_spark(spark) -> None:
    try:
        spark.stop()
        logger.info("SparkSession stopped.")
    except Exception:
        pass
