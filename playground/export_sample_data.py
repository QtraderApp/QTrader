import logging
from pathlib import Path
from typing import Optional

import duckdb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory configuration
DATA_DIR = Path("/home/javier/Data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet")
OUTPUT_DIR = Path("./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete")
TICKERS = ["AAPL"]
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"


class DuckDBClient:
    """DuckDB client for reading parquet data with partitioned structure."""

    def __init__(self, data_dir: Path, db_path: Optional[str] = None):
        self.data_dir = data_dir
        self.conn = duckdb.connect(db_path if db_path is not None else ":memory:")
        self.table_name = "market_data"

        # Configure DuckDB for better parquet performance
        self.conn.execute("SET threads TO 4")
        self.conn.execute("SET memory_limit = '4GB'")
        logger.info(f"Initialized DuckDB client with data directory: {data_dir}")

    def create_view(self, view_name: Optional[str] = None) -> bool:
        if view_name is None:
            view_name = self.table_name
        try:
            parquet_pattern = str(self.data_dir / "**" / "*.parquet")
            create_view_sql = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT * FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            """
            self.conn.execute(create_view_sql)
            logger.info(f"Created view '{view_name}' from parquet files")
            return True
        except Exception as e:
            logger.error(f"Error creating view: {e}")
            return False

    def export_sample_data(self, tickers: list, start_date: str, end_date: str, output_dir: Path) -> bool:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

            ticker_filter = "'" + "', '".join(tickers) + "'"
            export_query = f"""
            COPY (
                SELECT *
                FROM {self.table_name}
                WHERE Ticker IN ({ticker_filter})
                  AND TradeDate >= '{start_date}'
                  AND TradeDate <= '{end_date}'
                ORDER BY SecId, TradeDate
            ) TO '{output_dir}' (FORMAT PARQUET, PARTITION_BY (SecId), OVERWRITE_OR_IGNORE true)
            """

            logger.info(f"Exporting data for tickers: {tickers}")
            logger.info(f"Date range: {start_date} to {end_date}")
            self.conn.execute(export_query)
            logger.info("✓ Sample data exported successfully")
            return True
        except Exception as e:
            logger.error(f"Error exporting sample data: {e}")
            return False

    def check_tickers_exist(self, tickers: list) -> tuple[list, list]:
        """Check which specific tickers exist in the dataset. Returns (found, missing)."""
        try:
            ticker_filter = "'" + "', '".join(tickers) + "'"
            query = f"""
            SELECT DISTINCT Ticker
            FROM {self.table_name}
            WHERE Ticker IN ({ticker_filter})
            ORDER BY Ticker
            """
            result = self.conn.execute(query).fetchall()
            found_tickers = [row[0] for row in result]
            missing_tickers = [ticker for ticker in tickers if ticker not in found_tickers]
            return found_tickers, missing_tickers
        except Exception as e:
            logger.error(f"Error checking tickers: {e}")
            return [], tickers

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target tickers: {TICKERS}")
    print(f"Date range: {START_DATE} to {END_DATE}")

    client = DuckDBClient(DATA_DIR)

    print("\n=== Creating DuckDB View ===")
    if client.create_view():
        print("✓ View created successfully")

        print("\n=== Checking Target Tickers ===")
        found_tickers, missing_tickers = client.check_tickers_exist(TICKERS)

        print(f"Found tickers: {found_tickers}")
        if missing_tickers:
            print(f"Missing tickers: {missing_tickers}")
            logger.critical(f"CRITICAL ERROR: Required tickers not found in dataset: {missing_tickers}")
            print("✗ CRITICAL ERROR: Cannot proceed with missing tickers")
            client.close()
            exit(1)

        if found_tickers:
            print("\n=== Exporting Sample Data ===")
            export_success = client.export_sample_data(
                tickers=found_tickers, start_date=START_DATE, end_date=END_DATE, output_dir=OUTPUT_DIR
            )

            if export_success:
                print("✓ Sample data export completed successfully")

                print("\n=== Verifying Export ===")
                if OUTPUT_DIR.exists():
                    exported_files = list(OUTPUT_DIR.rglob("*.parquet"))
                    print(f"Exported {len(exported_files)} parquet files")

                    secid_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("SecId=")]
                    print(f"Created {len(secid_dirs)} SecId partitions")
                    for secid_dir in secid_dirs[:5]:
                        secid_files = list(secid_dir.glob("*.parquet"))
                        print(f"  {secid_dir.name}: {len(secid_files)} files")
            else:
                print("✗ Failed to export sample data")
        else:
            print("✗ No target tickers found in the dataset")
    else:
        print("✗ Failed to create view")

    client.close()


if __name__ == "__main__":
    main()
