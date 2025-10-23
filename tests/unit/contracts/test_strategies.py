"""
Unit tests for qtrader.contracts.strategies module.

Tests cover:
- SignalIntention enum values and behavior
- Signal model validation and immutability
- create_signal factory function
- Edge cases and error conditions
"""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from qtrader.contracts.strategies import CONTRACT_VERSION, Signal, SignalIntention, create_signal

# ============================================
# Fixtures
# ============================================


@pytest.fixture
def valid_timestamp():
    """Valid timestamp for signal creation."""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def minimal_signal_data(valid_timestamp):
    """Minimal valid data for Signal creation."""
    return {
        "timestamp": valid_timestamp,
        "strategy_id": "test_strategy",
        "symbol": "AAPL",
        "intention": SignalIntention.OPEN_LONG,
        "confidence": 0.75,
    }


@pytest.fixture
def full_signal_data(valid_timestamp):
    """Complete Signal data with all optional fields."""
    return {
        "timestamp": valid_timestamp,
        "strategy_id": "bb_breakout",
        "symbol": "TSLA",
        "intention": SignalIntention.OPEN_SHORT,
        "confidence": 0.85,
        "reason": "Oversold condition detected",
        "metadata": {
            "percent_b": -0.25,
            "bandwidth": 0.045,
            "price": 245.50,
        },
    }


# ============================================
# SignalIntention Tests
# ============================================


class TestSignalIntention:
    """Tests for SignalIntention enum."""

    def test_all_intentions_defined(self):
        """Test all four trading intentions are defined."""
        # Arrange & Act
        intentions = [e.value for e in SignalIntention]

        # Assert
        assert len(intentions) == 4
        assert "OPEN_LONG" in intentions
        assert "CLOSE_LONG" in intentions
        assert "OPEN_SHORT" in intentions
        assert "CLOSE_SHORT" in intentions

    def test_can_create_from_string(self):
        """Test SignalIntention can be created from string value."""
        # Arrange & Act
        intention = SignalIntention("OPEN_LONG")

        # Assert
        assert intention == SignalIntention.OPEN_LONG
        assert intention.value == "OPEN_LONG"

    def test_enum_equality(self):
        """Test enum instances are equal when same value."""
        # Arrange & Act
        intention1 = SignalIntention.OPEN_LONG
        intention2 = SignalIntention.OPEN_LONG

        # Assert
        assert intention1 == intention2
        assert intention1 is intention2  # Same singleton

    def test_invalid_intention_raises_error(self):
        """Test creating SignalIntention with invalid value raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="'INVALID' is not a valid SignalIntention"):
            SignalIntention("INVALID")

    @pytest.mark.parametrize(
        "intention,expected_value",
        [
            (SignalIntention.OPEN_LONG, "OPEN_LONG"),
            (SignalIntention.CLOSE_LONG, "CLOSE_LONG"),
            (SignalIntention.OPEN_SHORT, "OPEN_SHORT"),
            (SignalIntention.CLOSE_SHORT, "CLOSE_SHORT"),
        ],
    )
    def test_intention_value_matches_name(self, intention, expected_value):
        """Test each intention's value matches its name."""
        # Assert
        assert intention.value == expected_value


# ============================================
# Signal Model Tests - Happy Path
# ============================================


class TestSignalCreation:
    """Tests for Signal model creation and validation."""

    def test_create_signal_with_minimal_data(self, minimal_signal_data):
        """Test creating Signal with only required fields."""
        # Arrange & Act
        signal = Signal(**minimal_signal_data)

        # Assert
        assert signal.timestamp == minimal_signal_data["timestamp"]
        assert signal.strategy_id == "test_strategy"
        assert signal.symbol == "AAPL"
        assert signal.intention == SignalIntention.OPEN_LONG
        assert signal.confidence == 0.75
        assert signal.reason is None
        assert signal.metadata is None

    def test_create_signal_with_all_fields(self, full_signal_data):
        """Test creating Signal with all optional fields populated."""
        # Arrange & Act
        signal = Signal(**full_signal_data)

        # Assert
        assert signal.timestamp == full_signal_data["timestamp"]
        assert signal.strategy_id == "bb_breakout"
        assert signal.symbol == "TSLA"
        assert signal.intention == SignalIntention.OPEN_SHORT
        assert signal.confidence == 0.85
        assert signal.reason == "Oversold condition detected"
        assert signal.metadata == {
            "percent_b": -0.25,
            "bandwidth": 0.045,
            "price": 245.50,
        }

    def test_signal_is_immutable(self, minimal_signal_data):
        """Test Signal instances are frozen and cannot be modified."""
        # Arrange
        signal = Signal(**minimal_signal_data)

        # Act & Assert
        with pytest.raises(ValidationError, match="Instance is frozen"):
            signal.confidence = 0.99

        with pytest.raises(ValidationError, match="Instance is frozen"):
            signal.symbol = "MSFT"

    def test_signal_with_string_intention(self, valid_timestamp):
        """Test Signal accepts string intention and converts to enum."""
        # Arrange & Act
        # Test string coercion with valid enum value
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention="OPEN_LONG",  # type: ignore
            confidence=0.5,
        )

        # Assert
        assert signal.intention == SignalIntention.OPEN_LONG
        assert isinstance(signal.intention, SignalIntention)


# ============================================
# Signal Model Tests - Validation
# ============================================


class TestSignalValidation:
    """Tests for Signal model field validation."""

    def test_confidence_must_be_in_range_0_to_1(self, valid_timestamp):
        """Test confidence must be between 0.0 and 1.0 inclusive."""
        # Valid boundary values
        Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.0,  # Min valid
        )

        Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=1.0,  # Max valid
        )

        # Invalid: too low
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=-0.1,
            )

        # Invalid: too high
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=1.1,
            )

    def test_strategy_id_cannot_be_empty(self, valid_timestamp):
        """Test strategy_id must not be empty or whitespace."""
        # Empty string - fails min_length validation
        with pytest.raises(ValidationError, match="at least 1 character"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
            )

        # Whitespace only - fails custom validator
        with pytest.raises(ValidationError, match="cannot be empty"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="   ",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
            )

    def test_symbol_cannot_be_empty(self, valid_timestamp):
        """Test symbol must not be empty or whitespace."""
        # Empty string - fails min_length validation
        with pytest.raises(ValidationError, match="at least 1 character"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
            )

        # Whitespace only - fails custom validator
        with pytest.raises(ValidationError, match="cannot be empty"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="   ",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
            )

    def test_strategy_id_and_symbol_are_trimmed(self, valid_timestamp):
        """Test strategy_id and symbol have whitespace trimmed."""
        # Arrange & Act
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="  test_strategy  ",
            symbol="  AAPL  ",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
        )

        # Assert
        assert signal.strategy_id == "test_strategy"
        assert signal.symbol == "AAPL"

    def test_reason_max_length_500_chars(self, valid_timestamp):
        """Test reason field has max length of 500 characters."""
        # Valid: exactly 500 chars
        valid_reason = "x" * 500
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            reason=valid_reason,
        )
        assert signal.reason is not None
        assert len(signal.reason) == 500

        # Invalid: 501 chars
        with pytest.raises(ValidationError, match="at most 500 characters"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                reason="x" * 501,
            )

    def test_metadata_must_be_json_serializable(self, valid_timestamp):
        """Test metadata values must be JSON-serializable types."""
        # Valid: all allowed types
        valid_metadata = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
        }
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            metadata=valid_metadata,
        )
        assert signal.metadata == valid_metadata

        # Invalid: Decimal not allowed
        with pytest.raises(ValidationError, match="must be JSON-serializable"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                metadata={"decimal": Decimal("123.45")},
            )

        # Invalid: datetime not allowed
        with pytest.raises(ValidationError, match="must be JSON-serializable"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                metadata={"datetime": datetime.now()},
            )

        # Invalid: list not allowed
        with pytest.raises(ValidationError, match="must be JSON-serializable"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                metadata={"list": [1, 2, 3]},
            )

    def test_required_fields_validation(self, valid_timestamp):
        """Test all required fields must be provided."""
        # Missing timestamp
        with pytest.raises(ValidationError, match="timestamp"):
            Signal(
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                **{"timestamp": None},  # type: ignore
            )

        # Missing strategy_id
        with pytest.raises(ValidationError, match="strategy_id"):
            Signal(
                timestamp=valid_timestamp,
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                **{"strategy_id": None},  # type: ignore
            )

        # Missing symbol
        with pytest.raises(ValidationError, match="symbol"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                intention=SignalIntention.OPEN_LONG,
                confidence=0.5,
                **{"symbol": None},  # type: ignore
            )

        # Missing intention
        with pytest.raises(ValidationError, match="intention"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                confidence=0.5,
                **{"intention": None},  # type: ignore
            )

        # Missing confidence
        with pytest.raises(ValidationError, match="confidence"):
            Signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                **{"confidence": None},  # type: ignore
            )


# ============================================
# Signal Model Tests - Edge Cases
# ============================================


class TestSignalEdgeCases:
    """Tests for Signal model edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "confidence",
        [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0],
    )
    def test_confidence_boundary_values(self, valid_timestamp, confidence):
        """Test various valid confidence values across the range."""
        # Arrange & Act
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=confidence,
        )

        # Assert
        assert signal.confidence == confidence

    def test_special_characters_in_symbol(self, valid_timestamp):
        """Test symbols can contain special characters (e.g., futures, options)."""
        # Arrange & Act
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="SPY.US",  # Symbol with dot
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
        )

        # Assert
        assert signal.symbol == "SPY.US"

    def test_empty_metadata_dict_is_valid(self, valid_timestamp):
        """Test empty metadata dictionary is accepted."""
        # Arrange & Act
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            metadata={},
        )

        # Assert
        assert signal.metadata == {}

    def test_none_metadata_vs_empty_dict(self, valid_timestamp):
        """Test distinction between None and empty dict metadata."""
        # None metadata
        signal1 = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            metadata=None,
        )
        assert signal1.metadata is None

        # Empty dict metadata
        signal2 = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            metadata={},
        )
        assert signal2.metadata == {}

    def test_unicode_in_reason(self, valid_timestamp):
        """Test reason field can contain unicode characters."""
        # Arrange & Act
        signal = Signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.5,
            reason="Price ↑ above MA50 📈",
        )

        # Assert
        assert signal.reason == "Price ↑ above MA50 📈"


# ============================================
# create_signal Factory Function Tests
# ============================================


class TestCreateSignalFactory:
    """Tests for create_signal convenience factory function."""

    def test_create_signal_with_enum_intention(self, valid_timestamp):
        """Test create_signal works with SignalIntention enum."""
        # Arrange & Act
        signal = create_signal(
            timestamp=valid_timestamp,
            strategy_id="test_strategy",
            symbol="AAPL",
            intention=SignalIntention.OPEN_LONG,
            confidence=0.75,
        )

        # Assert
        assert isinstance(signal, Signal)
        assert signal.strategy_id == "test_strategy"
        assert signal.symbol == "AAPL"
        assert signal.intention == SignalIntention.OPEN_LONG
        assert signal.confidence == 0.75

    def test_create_signal_with_string_intention(self, valid_timestamp):
        """Test create_signal converts string intention to enum."""
        # Arrange & Act
        signal = create_signal(
            timestamp=valid_timestamp,
            strategy_id="test_strategy",
            symbol="AAPL",
            intention="OPEN_SHORT",  # String
            confidence=0.65,
        )

        # Assert
        assert signal.intention == SignalIntention.OPEN_SHORT
        assert isinstance(signal.intention, SignalIntention)

    def test_create_signal_with_all_parameters(self, valid_timestamp):
        """Test create_signal with all optional parameters."""
        # Arrange
        metadata = {"rsi": 72.5, "price": 150.25}

        # Act
        signal = create_signal(
            timestamp=valid_timestamp,
            strategy_id="rsi_strategy",
            symbol="TSLA",
            intention="CLOSE_LONG",
            confidence=0.90,
            reason="RSI overbought",
            metadata=metadata,
        )

        # Assert
        assert signal.strategy_id == "rsi_strategy"
        assert signal.symbol == "TSLA"
        assert signal.intention == SignalIntention.CLOSE_LONG
        assert signal.confidence == 0.90
        assert signal.reason == "RSI overbought"
        assert signal.metadata == metadata

    def test_create_signal_invalid_string_intention(self, valid_timestamp):
        """Test create_signal raises error for invalid string intention."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="'INVALID' is not a valid SignalIntention"):
            create_signal(
                timestamp=valid_timestamp,
                strategy_id="test",
                symbol="AAPL",
                intention="INVALID",
                confidence=0.5,
            )

    @pytest.mark.parametrize(
        "intention_str,expected_enum",
        [
            ("OPEN_LONG", SignalIntention.OPEN_LONG),
            ("CLOSE_LONG", SignalIntention.CLOSE_LONG),
            ("OPEN_SHORT", SignalIntention.OPEN_SHORT),
            ("CLOSE_SHORT", SignalIntention.CLOSE_SHORT),
        ],
    )
    def test_create_signal_all_intentions(self, valid_timestamp, intention_str, expected_enum):
        """Test create_signal handles all intention strings correctly."""
        # Arrange & Act
        signal = create_signal(
            timestamp=valid_timestamp,
            strategy_id="test",
            symbol="AAPL",
            intention=intention_str,
            confidence=0.5,
        )

        # Assert
        assert signal.intention == expected_enum


# ============================================
# Contract Version Tests
# ============================================


class TestContractVersion:
    """Tests for contract version constant."""

    def test_contract_version_is_defined(self):
        """Test CONTRACT_VERSION constant is defined."""
        assert CONTRACT_VERSION is not None

    def test_contract_version_format(self):
        """Test CONTRACT_VERSION follows semver format."""
        # Should be in format "X.Y.Z"
        parts = CONTRACT_VERSION.split(".")
        assert len(parts) == 3, "Version should have 3 parts (major.minor.patch)"
        assert all(part.isdigit() for part in parts), "All version parts should be numeric"

    def test_contract_version_value(self):
        """Test CONTRACT_VERSION has expected value."""
        assert CONTRACT_VERSION == "1.0.0"


# ============================================
# Public API Tests
# ============================================


class TestPublicAPI:
    """Tests for module's public API exports."""

    def test_all_exports_are_defined(self):
        """Test __all__ contains expected exports."""
        from qtrader.contracts.strategies import __all__

        expected = [
            "CONTRACT_VERSION",
            "Signal",
            "SignalIntention",
            "create_signal",
        ]

        assert set(__all__) == set(expected)

    def test_can_import_all_exports(self):
        """Test all exported items can be imported."""
        from qtrader.contracts.strategies import CONTRACT_VERSION, Signal, SignalIntention, create_signal

        # All should be importable without error
        assert CONTRACT_VERSION is not None
        assert Signal is not None
        assert SignalIntention is not None
        assert create_signal is not None
