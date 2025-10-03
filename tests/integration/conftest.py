"""
Shared fixtures for integration tests.

These fixtures provide real data and components configured for
end-to-end testing scenarios.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
import pytz

from qtrader.execution.config import ExecutionConfig
from qtrader.execution.engine import ExecutionEngine
from qtrader.models.portfolio import Portfolio

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def data_dir():
    """Path to fixture data directory."""
    return Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def portfolio_100k():
    """Portfolio with $100k starting capital."""
    return Portfolio(initial_cash=Decimal("100000.00"))


@pytest.fixture
def portfolio_1m():
    """Portfolio with $1M starting capital."""
    return Portfolio(initial_cash=Decimal("1000000.00"))


@pytest.fixture
def portfolio_10m():
    """Portfolio with $10M starting capital."""
    return Portfolio(initial_cash=Decimal("10000000.00"))


@pytest.fixture
def execution_config_default():
    """Default execution configuration."""
    return ExecutionConfig()


@pytest.fixture
def execution_config_conservative():
    """Conservative execution configuration (low participation)."""
    return ExecutionConfig(
        max_participation=Decimal("0.05"),  # 5% of volume
        queue_bars=2,
        allow_high_participation=False,
    )


@pytest.fixture
def execution_config_aggressive():
    """Aggressive execution configuration (high participation allowed)."""
    return ExecutionConfig(
        max_participation=Decimal("0.25"),  # 25% of volume
        queue_bars=5,
        allow_high_participation=True,
    )


@pytest.fixture
def engine_100k(portfolio_100k, execution_config_default):
    """ExecutionEngine with $100k portfolio and default config."""
    return ExecutionEngine(
        portfolio=portfolio_100k,
        config=execution_config_default,
    )


@pytest.fixture
def engine_1m(portfolio_1m, execution_config_default):
    """ExecutionEngine with $1M portfolio and default config."""
    return ExecutionEngine(
        portfolio=portfolio_1m,
        config=execution_config_default,
    )


@pytest.fixture
def engine_10m(portfolio_10m, execution_config_default):
    """ExecutionEngine with $10M portfolio and default config."""
    return ExecutionEngine(
        portfolio=portfolio_10m,
        config=execution_config_default,
    )


@pytest.fixture
def test_dates_2020_jan():
    """Common test dates for January 2020."""
    return {
        "start": datetime(2020, 1, 2, tzinfo=ET),
        "end": datetime(2020, 1, 31, tzinfo=ET),
        "mid": datetime(2020, 1, 15, tzinfo=ET),
    }


@pytest.fixture
def test_symbols_single():
    """Single symbol for testing."""
    return ["AAPL"]


@pytest.fixture
def test_symbols_multi():
    """Multiple symbols for testing."""
    return ["AAPL", "MSFT", "AMZN"]
