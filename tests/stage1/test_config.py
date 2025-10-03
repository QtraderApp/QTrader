"""Tests for data configuration."""

from qtrader.config.data_config import (
    DataConfig,
    ValidationConfig,
    BarSchemaConfig,
    AdjustmentSchemaConfig,
)


def test_validation_config_defaults():
    """ValidationConfig should have sensible defaults."""
    config = ValidationConfig()
    assert config.epsilon == 0.0
    assert config.ohlc_policy == "strict_raise"
    assert config.close_only_fields == ["close"]


def test_bar_schema_config_creation():
    """BarSchemaConfig should map vendor columns to Bar fields."""
    schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="MarketHoursVolume",
    )
    assert schema.ts == "TradeDate"
    assert schema.symbol == "Ticker"
    assert schema.volume == "MarketHoursVolume"


def test_adjustment_schema_config_creation():
    """AdjustmentSchemaConfig should map vendor columns to AdjustmentEvent fields."""
    schema = AdjustmentSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        event_type="AdjustmentReason",
        px_factor="CumulativePriceFactor",
        vol_factor="CumulativeVolumeFactor",
        metadata_fields=["AdjustmentFactor"],
    )
    assert schema.ts == "TradeDate"
    assert schema.event_type == "AdjustmentReason"
    assert schema.metadata_fields == ["AdjustmentFactor"]


def test_data_config_defaults():
    """DataConfig should have sensible defaults."""
    # Need bar_schema as required field
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    config = DataConfig(bar_schema=bar_schema)
    
    assert config.mode == "adjusted"
    assert config.frequency == "1d"
    assert config.timezone == "America/New_York"
    assert config.strict_frequency is True
    assert config.decimals == {"price": 4, "cash": 4}
    assert config.validation.ohlc_policy == "strict_raise"


def test_data_config_with_adjustment_schema():
    """DataConfig should accept optional adjustment_schema."""
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    adj_schema = AdjustmentSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        event_type="AdjustmentReason",
        px_factor="CumulativePriceFactor",
        vol_factor="CumulativeVolumeFactor",
    )
    config = DataConfig(bar_schema=bar_schema, adjustment_schema=adj_schema)
    
    assert config.adjustment_schema is not None
    assert config.adjustment_schema.event_type == "AdjustmentReason"
