"""Unit tests for base Indicator class."""

import pytest

from qtrader.api import Context
from qtrader.indicators.base import Indicator
from qtrader.models.bar import Bar


class SimpleIndicator(Indicator[float]):
    """Test indicator that returns the close price."""

    def __init__(self):
        super().__init__()
        self.compute_count = 0

    def compute(self, symbol: str, ctx: Context) -> float | None:
        """Return close price of current bar."""
        self.compute_count += 1
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None
        return float(bars[-1].close)


def test_indicator_initialization():
    """Test indicator initializes with empty cache."""
    ind = SimpleIndicator()
    assert ind._cache == {}
    assert ind._bar_counter == {}
    assert ind.compute_count == 0


def test_indicator_compute_called():
    """Test compute method is called correctly."""
    from datetime import datetime, timezone
    from decimal import Decimal

    ind = SimpleIndicator()
    ctx = Context()

    # Add bar to context
    bar = Bar(
        ts=datetime.now(timezone.utc),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=1000,
    )
    ctx._add_bar_to_history(bar)

    # Compute indicator
    result = ind.compute("TEST", ctx)
    assert result == 103.0
    assert ind.compute_count == 1


def test_indicator_reset_clears_cache():
    """Test reset clears cached data for symbol."""
    from datetime import datetime, timezone
    from decimal import Decimal

    ind = SimpleIndicator()
    ctx = Context()

    bar = Bar(
        ts=datetime.now(timezone.utc),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=1000,
    )
    ctx._add_bar_to_history(bar)

    # Compute to populate cache
    ind.compute("TEST", ctx)
    ind._cache[("TEST", 0)] = 103.0
    ind._bar_counter["TEST"] = 1

    # Reset
    ind.reset("TEST")

    # Cache should be cleared
    assert ("TEST", 0) not in ind._cache
    assert "TEST" not in ind._bar_counter


def test_indicator_multiple_symbols():
    """Test indicator handles multiple symbols independently."""
    from datetime import datetime, timezone
    from decimal import Decimal

    ind = SimpleIndicator()
    ctx = Context()

    # Add bars for two symbols
    bar1 = Bar(
        ts=datetime.now(timezone.utc),
        symbol="AAPL",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("103"),
        volume=1000,
    )
    bar2 = Bar(
        ts=datetime.now(timezone.utc),
        symbol="MSFT",
        open=Decimal("200"),
        high=Decimal("205"),
        low=Decimal("199"),
        close=Decimal("203"),
        volume=2000,
    )

    ctx._add_bar_to_history(bar1)
    ctx._add_bar_to_history(bar2)

    # Compute for both
    result1 = ind.compute("AAPL", ctx)
    result2 = ind.compute("MSFT", ctx)

    assert result1 == 103.0
    assert result2 == 203.0


def test_indicator_warmup_default_implementation():
    """Test warmup has default no-op implementation."""
    ind = SimpleIndicator()
    ctx = Context()

    # Should not raise
    ind.warmup("TEST", ctx)


def test_indicator_abstract_compute_enforcement():
    """Test that Indicator requires compute implementation."""
    from abc import ABC

    # Indicator should be abstract
    assert issubclass(Indicator, ABC)

    # Cannot instantiate without compute
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):

        class IncompleteIndicator(Indicator[float]):
            pass

        IncompleteIndicator()  # type: ignore
