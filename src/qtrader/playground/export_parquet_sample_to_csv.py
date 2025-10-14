import logging
from pathlib import Path
from typing import Optional

import duckdb

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PARQUET_DIR = REPO_ROOT / "data" / "us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete"
OUTPUT_CSV_DIR = REPO_ROOT / "data" / "csv"


def export_sample_to_csv(sample_dir: Path, out_dir: Path, db_path: Optional[str] = None) -> None:
    """Export the repo's sample parquet dataset into one CSV per SecId under out_dir.

    - Reads all parquet files under sample_dir with hive_partitioning=true
    - Finds distinct SecIds
    - For each SecId, exports a CSV named secid_<SecId>.csv in out_dir
    """
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample parquet directory not found: {sample_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {out_dir}")

    con = duckdb.connect(db_path if db_path is not None else ":memory:")
    try:
        parquet_glob = str(sample_dir / "**" / "*.parquet")
        con.execute(
            f"""
            CREATE OR REPLACE VIEW sample AS
            SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)
            """
        )
        logger.info("View 'sample' created over parquet files")

        secids = [row[0] for row in con.execute("SELECT DISTINCT SecId FROM sample ORDER BY SecId").fetchall()]
        if not secids:
            logger.warning("No SecIds found in sample dataset; nothing to export")
            return

        logger.info(f"Found {len(secids)} SecIds: {secids}")
        for secid in secids:
            out_file = out_dir / f"secid_{secid}.csv"
            logger.info(f"Exporting SecId={secid} -> {out_file}")
            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM sample
                    WHERE SecId = {secid}
                    ORDER BY TradeDate
                ) TO '{out_file}' WITH (FORMAT CSV, HEADER TRUE, OVERWRITE_OR_IGNORE TRUE)
                """
            )
        logger.info("All SecId CSV exports completed successfully")
    finally:
        con.close()


if __name__ == "__main__":
    print(f"Sample parquet dir: {SAMPLE_PARQUET_DIR}")
    print(f"CSV output dir:     {OUTPUT_CSV_DIR}")
    export_sample_to_csv(SAMPLE_PARQUET_DIR, OUTPUT_CSV_DIR)
