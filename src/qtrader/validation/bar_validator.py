"""Validates Bar integrity and applies OHLC policies."""

from datetime import timedelta
from decimal import Decimal
from typing import Optional

import structlog

from qtrader.config.data_config import DataConfig
from qtrader.models.bar import Bar, OHLCPolicy

logger = structlog.get_logger()


class BarValidator:
    """
    Validates Bar integrity and applies OHLC policies.

    Tracks statistics on malformed bars for inclusion in run.json.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.epsilon = Decimal(str(config.validation.epsilon))
        self.policy = OHLCPolicy(config.validation.ohlc_policy)
        self.malformed_count = 0
        self.skipped_count = 0
        self.close_only_count = 0
        self.malformed_samples: list[tuple[str, str]] = []  # (date, symbol) for first/last 10

    def validate_ohlc(self, bar: Bar) -> tuple[bool, Optional[str]]:
        """
        Validate OHLC relationships.

        Returns:
            (is_valid, reason) where reason is None if valid
        """
        # Check high >= max(open, close)
        if bar.high < max(bar.open, bar.close) - self.epsilon:
            return False, f"high ({bar.high}) < max(open={bar.open}, close={bar.close})"

        # Check low <= min(open, close)
        if bar.low > min(bar.open, bar.close) + self.epsilon:
            return False, f"low ({bar.low}) > min(open={bar.open}, close={bar.close})"

        # Check low <= high
        if bar.low > bar.high + self.epsilon:
            return False, f"low ({bar.low}) > high ({bar.high})"

        # Check volume >= 0
        if bar.volume < 0:
            return False, f"volume ({bar.volume}) < 0"

        return True, None

    def process_bar(self, bar: Bar) -> tuple[Optional[Bar], bool]:
        """
        Process bar according to OHLC policy.

        Returns:
            (bar_or_none, is_close_only)
            - bar_or_none: None if skipped, otherwise the bar
            - is_close_only: True if bar should only use close (no limit/stop)
        """
        is_valid, reason = self.validate_ohlc(bar)

        if is_valid:
            return bar, False

        # Malformed bar - apply policy
        self.malformed_count += 1

        # Store sample for reporting (first/last 10)
        if len(self.malformed_samples) < 10 or self.malformed_count > (self.malformed_count - 10):
            self.malformed_samples.append((bar.ts.strftime("%Y-%m-%d"), bar.symbol))
            if len(self.malformed_samples) > 20:  # Keep first 10 and last 10
                self.malformed_samples = self.malformed_samples[:10] + self.malformed_samples[-10:]

        if self.policy == OHLCPolicy.STRICT_RAISE:
            logger.error("bar_validator.malformed_strict", symbol=bar.symbol, ts=bar.ts, reason=reason)
            raise ValueError(f"Malformed OHLC bar at {bar.ts} for {bar.symbol}: {reason}")

        elif self.policy == OHLCPolicy.WARN_SKIP_BAR:
            logger.warning("bar_validator.malformed_skip", symbol=bar.symbol, ts=bar.ts, reason=reason)
            self.skipped_count += 1
            return None, False

        elif self.policy == OHLCPolicy.WARN_USE_CLOSE_ONLY:
            logger.warning("bar_validator.malformed_close_only", symbol=bar.symbol, ts=bar.ts, reason=reason)
            self.close_only_count += 1
            return bar, True

        # Unreachable: all enum values handled above
        # This case exists for defensive programming and type exhaustiveness
        raise AssertionError(f"Unhandled OHLCPolicy: {self.policy}")  # pragma: no cover

    def validate_frequency(self, bars: list[Bar]) -> bool:
        """
        Validate that bars match expected frequency.

        Returns True if valid, raises ValueError if strict_frequency=true and invalid.
        """
        if not self.config.strict_frequency or len(bars) < 3:
            return True

        # Calculate median delta per symbol
        deltas = []
        for i in range(1, len(bars)):
            if bars[i].symbol == bars[i - 1].symbol:
                delta = bars[i].ts - bars[i - 1].ts
                deltas.append(delta)

        if not deltas:
            return True

        deltas.sort()
        median_delta = deltas[len(deltas) // 2]

        # Expected delta for frequency
        freq = self.config.frequency
        expected_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }

        expected = expected_deltas.get(freq)
        if expected and abs((median_delta - expected).total_seconds()) > 3600:  # 1 hour tolerance
            msg = f"Frequency mismatch: expected {freq}, median delta {median_delta}"
            # Handle based on strict_frequency setting
            strict = self.config.strict_frequency
            if strict:
                logger.error("bar_validator.frequency_mismatch", expected=freq, median=str(median_delta))
                raise ValueError(msg)
            logger.warning("bar_validator.frequency_mismatch", expected=freq, median=str(median_delta))  # type: ignore[unreachable]
            return False

        logger.info("bar_validator.frequency_validated", expected=freq, median=str(median_delta))
        return True

    def get_stats(self) -> dict:
        """Return validation statistics for run.json."""
        return {
            "malformed_bars": self.malformed_count,
            "skipped": self.skipped_count,
            "close_only": self.close_only_count,
            "malformed_samples": self.malformed_samples,
        }
