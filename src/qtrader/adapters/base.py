"""Protocol for data adapters that normalize vendor data to canonical Bar."""

from typing import Iterator, Protocol

from qtrader.config.data_config import DataConfig
from qtrader.models.bar import AdjustmentEvent, Bar, DataMode


class DataAdapter(Protocol):
    """
    Protocol for data adapters that normalize vendor data to canonical Bar.

    Responsibilities:
    1. Convert vendor schema → canonical Bar (OHLCV only)
    2. Optionally extract adjustment metadata → AdjustmentEvent
    3. Declare DataMode (adjusted, unadjusted, split_adjusted)

    New vendor = new DataAdapter + schema config + unit tests; engine unchanged.
    """

    def can_read(self) -> bool:
        """Check if this adapter can read from the configured source."""
        ...

    def schema_version(self) -> str:
        """
        Return the adapter schema version for reproducibility.

        This version is persisted in run.json for audit trail.
        Example: "algoseek-parquet-v1.0", "iqfeed-tick-v2.1"
        """
        ...

    def get_data_mode(self) -> DataMode:
        """
        Declare if OHLCV prices are adjusted or unadjusted.

        Critical for execution engine to interpret prices correctly.
        """
        ...

    def read_bars(self, config: DataConfig) -> Iterator[Bar]:
        """
        Read and normalize data to canonical Bar objects.

        Pipeline: Read RawRecord → Map columns → Convert types → Validate → Emit Bar

        Args:
            config: Data configuration including bar schema and timezone

        Yields:
            Bar objects with:
            - Decimal prices (quantized to config.decimals.price)
            - Timezone-aware timestamps
            - Validated OHLC relationships (high >= max(o,c), low <= min(o,c))
        """
        ...

    def read_adjustments(self, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata (optional).

        Args:
            config: Data configuration including adjustment schema

        Returns:
            Empty iterator if:
            - Data is unadjusted with no adjustment table
            - Vendor doesn't provide adjustment metadata

        Yields:
            AdjustmentEvent objects with corporate action details
        """
        ...
