"""Tests for Bar validator."""

import pytz
from datetime import datetime
from decimal import Decimal

import pytest

from qtrader.config.data_config import DataConfig, ValidationConfig, BarSchemaConfig
from qtrader.models.bar import Bar
from qtrader.validation.bar_validator import BarValidator


@pytest.fixture
def good_bar():
    """Create a valid bar."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("102"),
        volume=1000000,
    )


@pytest.fixture
def bad_bar_high_low():
    """Create bar with high < open."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("99"),  # Invalid: high < open
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1000,
    )


@pytest.fixture
def bad_bar_low_high():
    """Create bar with low > close."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("103"),  # Invalid: low > close
        close=Decimal("102"),
        volume=1000,
    )


@pytest.fixture
def data_config_strict():
    """Create data config with strict policy."""
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    return DataConfig(
        bar_schema=bar_schema,
        validation=ValidationConfig(ohlc_policy="strict_raise"),
    )


@pytest.fixture
def data_config_skip():
    """Create data config with skip policy."""
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    return DataConfig(
        bar_schema=bar_schema,
        validation=ValidationConfig(ohlc_policy="warn_skip_bar"),
    )


@pytest.fixture
def data_config_close_only():
    """Create data config with close-only policy."""
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    return DataConfig(
        bar_schema=bar_schema,
        validation=ValidationConfig(ohlc_policy="warn_use_close_only"),
    )


def test_validator_accepts_good_bar(good_bar, data_config_strict):
    """Validator should accept valid bars."""
    validator = BarValidator(data_config_strict)

    is_valid, reason = validator.validate_ohlc(good_bar)
    assert is_valid is True
    assert reason is None


def test_validator_detects_bad_ohlc(bad_bar_high_low, data_config_strict):
    """Validator should detect malformed OHLC."""
    validator = BarValidator(data_config_strict)

    is_valid, reason = validator.validate_ohlc(bad_bar_high_low)
    assert is_valid is False
    assert "high" in reason


def test_validator_strict_raise_policy(bad_bar_high_low, data_config_strict):
    """Validator should raise on malformed bar with strict policy."""
    validator = BarValidator(data_config_strict)

    with pytest.raises(ValueError, match="Malformed OHLC"):
        validator.process_bar(bad_bar_high_low)


def test_validator_warn_skip_bar_policy(bad_bar_high_low, data_config_skip):
    """Validator should skip bar with warn_skip_bar policy."""
    validator = BarValidator(data_config_skip)

    result, is_close_only = validator.process_bar(bad_bar_high_low)
    assert result is None
    assert is_close_only is False
    assert validator.skipped_count == 1


def test_validator_warn_use_close_only_policy(bad_bar_high_low, data_config_close_only):
    """Validator should allow bar but flag as close-only."""
    validator = BarValidator(data_config_close_only)

    result, is_close_only = validator.process_bar(bad_bar_high_low)
    assert result is not None
    assert is_close_only is True
    assert validator.close_only_count == 1


def test_validator_tracks_statistics(bad_bar_high_low, bad_bar_low_high, data_config_skip):
    """Validator should track statistics for run.json."""
    validator = BarValidator(data_config_skip)

    validator.process_bar(bad_bar_high_low)
    validator.process_bar(bad_bar_low_high)

    stats = validator.get_stats()
    assert stats["malformed_bars"] == 2
    assert stats["skipped"] == 2
    assert len(stats["malformed_samples"]) == 2


def test_validator_processes_good_bar_without_modification(good_bar, data_config_strict):
    """Validator should pass through good bars unchanged."""
    validator = BarValidator(data_config_strict)

    result, is_close_only = validator.process_bar(good_bar)
    assert result == good_bar
    assert is_close_only is False
    assert validator.malformed_count == 0
