"""Unit tests for qtrader.adapters.adjustments module.

Tests the generic adjustment calculation utilities for converting between
different price series types (unadjusted, capital-adjusted, total-return).
"""

from decimal import Decimal

from qtrader.adapters.adjustments import (
    compute_capital_adjusted_price,
    compute_unadjusted_price,
    compute_unadjusted_volume,
)


class TestComputeUnadjustedPrice:
    """Tests for compute_unadjusted_price function."""

    def test_compute_unadjusted_price_basic_multiplication(self):
        """Test basic price unadjustment calculation."""
        # Arrange
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("2.0")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        assert result == Decimal("200.00"), "Unadjusted price should be adjusted * factor"

    def test_compute_unadjusted_price_real_world_apple_example(self):
        """Test using real AAPL data from docstring example."""
        # Arrange: AAPL on 2019-01-02
        adjusted_price = Decimal("157.92")
        cumulative_price_factor = Decimal("7.925959")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        # 157.92 * 7.925959 = 1251.66744528
        expected = Decimal("1251.66744528")
        assert result == expected, "Should match real AAPL historical data"

    def test_compute_unadjusted_price_no_adjustment_factor_one(self):
        """Test when cumulative factor is 1.0 (no adjustments)."""
        # Arrange
        adjusted_price = Decimal("50.25")
        cumulative_price_factor = Decimal("1.0")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        assert result == adjusted_price, "Factor of 1.0 should return unchanged price"

    def test_compute_unadjusted_price_very_small_factor(self):
        """Test with very small cumulative factor (heavily adjusted down)."""
        # Arrange
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("0.001")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        assert result == Decimal("0.1"), "Should handle small factors correctly"

    def test_compute_unadjusted_price_very_large_factor(self):
        """Test with very large cumulative factor (many splits)."""
        # Arrange
        adjusted_price = Decimal("25.50")
        cumulative_price_factor = Decimal("1000.0")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        assert result == Decimal("25500.0"), "Should handle large factors correctly"

    def test_compute_unadjusted_price_high_precision(self):
        """Test that decimal precision is maintained."""
        # Arrange
        adjusted_price = Decimal("123.456789")
        cumulative_price_factor = Decimal("2.718281828")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        # 123.456789 * 2.718281828 = 335.590346081930292
        # Verify precision is maintained (not rounded prematurely)
        assert str(result).startswith("335.59"), "Should maintain decimal precision"
        assert len(str(result).replace(".", "")) > 10, "Should preserve significant digits"

    def test_compute_unadjusted_price_zero_price(self):
        """Test edge case with zero adjusted price."""
        # Arrange
        adjusted_price = Decimal("0.0")
        cumulative_price_factor = Decimal("5.0")

        # Act
        result = compute_unadjusted_price(adjusted_price, cumulative_price_factor)

        # Assert
        assert result == Decimal("0.0"), "Zero price should remain zero"


class TestComputeCapitalAdjustedPrice:
    """Tests for compute_capital_adjusted_price function."""

    def test_compute_capital_adjusted_price_removes_dividend_adjustment(self):
        """Test that capital adjustment removes dividend component."""
        # Arrange
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("10.0")  # Includes splits AND dividends
        cumulative_volume_factor = Decimal("5.0")  # Only splits

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Ratio 10.0/5.0 = 2.0 is the dividend factor
        # 100.00 * 2.0 = 200.00 (splits only, no dividends)
        assert result == Decimal("200.00"), "Should isolate split adjustments only"

    def test_compute_capital_adjusted_price_real_world_apple_example(self):
        """Test using real AAPL data from docstring example."""
        # Arrange: AAPL on 2019-01-02
        adjusted_price = Decimal("157.92")
        cumulative_price_factor = Decimal("7.925959")
        cumulative_volume_factor = Decimal("7.0")

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Dividend factor = 7.925959 / 7.0 = 1.132279857142857142857142857
        # 157.92 * 1.132279857142857... = 178.80963504
        expected_approx = Decimal("178.81")
        assert abs(result - expected_approx) < Decimal("0.01"), "Should match real AAPL capital-adjusted price"

    def test_compute_capital_adjusted_price_equal_factors_no_dividends(self):
        """Test when price and volume factors are equal (no dividends paid)."""
        # Arrange
        adjusted_price = Decimal("50.00")
        cumulative_price_factor = Decimal("4.0")
        cumulative_volume_factor = Decimal("4.0")

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Ratio 4.0/4.0 = 1.0, so price unchanged
        assert result == adjusted_price, "Equal factors means no dividend adjustments"

    def test_compute_capital_adjusted_price_no_adjustments_factor_one(self):
        """Test when both factors are 1.0 (no adjustments at all)."""
        # Arrange
        adjusted_price = Decimal("75.50")
        cumulative_price_factor = Decimal("1.0")
        cumulative_volume_factor = Decimal("1.0")

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        assert result == adjusted_price, "Factors of 1.0 should return unchanged price"

    def test_compute_capital_adjusted_price_high_dividend_component(self):
        """Test with significant dividend adjustments."""
        # Arrange
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("20.0")  # Large dividend impact
        cumulative_volume_factor = Decimal("2.0")  # Small split impact

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Ratio 20.0/2.0 = 10.0 (large dividend factor)
        assert result == Decimal("1000.00"), "Should handle high dividend adjustments"

    def test_compute_capital_adjusted_price_precision_maintained(self):
        """Test that decimal precision is maintained through calculation."""
        # Arrange
        adjusted_price = Decimal("123.456789")
        cumulative_price_factor = Decimal("3.141592653589793")
        cumulative_volume_factor = Decimal("2.718281828459045")

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Verify calculation maintains precision
        dividend_factor = cumulative_price_factor / cumulative_volume_factor
        expected = adjusted_price * dividend_factor
        assert result == expected, "Should maintain full decimal precision"

    def test_compute_capital_adjusted_price_volume_factor_greater_than_price(self):
        """Test when volume factor exceeds price factor (unusual but valid)."""
        # Arrange
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("5.0")
        cumulative_volume_factor = Decimal("10.0")

        # Act
        result = compute_capital_adjusted_price(adjusted_price, cumulative_price_factor, cumulative_volume_factor)

        # Assert
        # Ratio 5.0/10.0 = 0.5
        assert result == Decimal("50.00"), "Should handle ratio < 1.0"


class TestComputeUnadjustedVolume:
    """Tests for compute_unadjusted_volume function."""

    def test_compute_unadjusted_volume_basic_division(self):
        """Test basic volume unadjustment calculation."""
        # Arrange
        adjusted_volume = 1000
        cumulative_volume_factor = Decimal("2.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert result == 500, "Unadjusted volume should be adjusted / factor"
        assert isinstance(result, int), "Result should be integer type"

    def test_compute_unadjusted_volume_real_world_apple_example(self):
        """Test using real AAPL data from docstring example."""
        # Arrange: AAPL on 2019-01-02
        adjusted_volume = 30606605
        cumulative_volume_factor = Decimal("7.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        # 30606605 / 7.0 = 4372372.142857... → rounds to 4372372
        assert result == 4372372, "Should match real AAPL historical volume"

    def test_compute_unadjusted_volume_no_adjustment_factor_one(self):
        """Test when cumulative factor is 1.0 (no adjustments)."""
        # Arrange
        adjusted_volume = 5000000
        cumulative_volume_factor = Decimal("1.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert result == adjusted_volume, "Factor of 1.0 should return unchanged volume"

    def test_compute_unadjusted_volume_rounding_half_up(self):
        """Test that ROUND_HALF_UP is applied correctly."""
        # Arrange
        adjusted_volume = 1000
        cumulative_volume_factor = Decimal("3.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        # 1000 / 3.0 = 333.333... → rounds to 333
        assert result == 333, "Should round down from .333"

    def test_compute_unadjusted_volume_rounding_half_up_exactly_half(self):
        """Test rounding behavior at exactly 0.5."""
        # Arrange
        adjusted_volume = 15
        cumulative_volume_factor = Decimal("2.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        # 15 / 2.0 = 7.5 → rounds up to 8 (ROUND_HALF_UP)
        assert result == 8, "Should round up from exactly .5"

    def test_compute_unadjusted_volume_large_split_factor(self):
        """Test with large cumulative factor (multiple splits)."""
        # Arrange
        adjusted_volume = 10000000
        cumulative_volume_factor = Decimal("100.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert result == 100000, "Should handle large factors correctly"

    def test_compute_unadjusted_volume_small_volume(self):
        """Test with very small volume numbers."""
        # Arrange
        adjusted_volume = 100
        cumulative_volume_factor = Decimal("2.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert result == 50, "Should handle small volumes correctly"

    def test_compute_unadjusted_volume_zero_volume(self):
        """Test edge case with zero volume."""
        # Arrange
        adjusted_volume = 0
        cumulative_volume_factor = Decimal("5.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert result == 0, "Zero volume should remain zero"
        assert isinstance(result, int), "Result should be integer type"

    def test_compute_unadjusted_volume_precise_factor(self):
        """Test with non-round cumulative factor."""
        # Arrange
        adjusted_volume = 1000000
        cumulative_volume_factor = Decimal("2.5")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        # 1000000 / 2.5 = 400000
        assert result == 400000, "Should handle non-integer factors"

    def test_compute_unadjusted_volume_complex_rounding_case(self):
        """Test rounding with complex division result."""
        # Arrange
        adjusted_volume = 999999
        cumulative_volume_factor = Decimal("7.0")

        # Act
        result = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        # 999999 / 7.0 = 142857.0 (exact)
        assert result == 142857, "Should handle complex division correctly"


class TestIntegrationScenarios:
    """Integration tests showing typical usage patterns."""

    def test_full_adjustment_workflow_splits_and_dividends(self):
        """Test converting between all three price series types."""
        # Arrange: A stock with 2:1 split and dividends
        adjusted_price = Decimal("100.00")
        cumulative_price_factor = Decimal("3.0")  # 2.0 split + 1.5x dividends
        cumulative_volume_factor = Decimal("2.0")  # 2:1 split only

        # Act: Compute all price types
        unadjusted = compute_unadjusted_price(adjusted_price, cumulative_price_factor)
        capital_adjusted = compute_capital_adjusted_price(
            adjusted_price, cumulative_price_factor, cumulative_volume_factor
        )

        # Assert: Relationships between price types
        assert unadjusted == Decimal("300.00"), "Unadjusted = adjusted * 3.0"
        assert capital_adjusted == Decimal("150.00"), "Capital = adjusted * (3.0/2.0)"
        assert adjusted_price < capital_adjusted < unadjusted, "Price hierarchy: adjusted < capital < unadjusted"

    def test_no_corporate_actions_all_prices_equal(self):
        """Test that all price types are equal when no adjustments exist."""
        # Arrange: No splits or dividends (both factors = 1.0)
        adjusted_price = Decimal("50.00")
        cumulative_price_factor = Decimal("1.0")
        cumulative_volume_factor = Decimal("1.0")

        # Act
        unadjusted = compute_unadjusted_price(adjusted_price, cumulative_price_factor)
        capital_adjusted = compute_capital_adjusted_price(
            adjusted_price, cumulative_price_factor, cumulative_volume_factor
        )

        # Assert: All prices should be identical
        assert unadjusted == adjusted_price == capital_adjusted, "No adjustments means all price types equal"

    def test_volume_adjustment_consistency_with_split(self):
        """Test that volume adjustment is consistent with price split adjustment."""
        # Arrange: 4:1 stock split (volume factor = 4.0)
        adjusted_volume = 4000000
        cumulative_volume_factor = Decimal("4.0")

        # Act
        unadjusted_vol = compute_unadjusted_volume(adjusted_volume, cumulative_volume_factor)

        # Assert
        assert unadjusted_vol == 1000000, "4:1 split means volume divided by 4"
        # Verify: pre-split price * pre-split volume ≈ post-split price * post-split volume
        # (This is just demonstrating the relationship, not testing the function)
        split_ratio = float(cumulative_volume_factor)
        assert adjusted_volume == unadjusted_vol * split_ratio, "Adjusted volume = unadjusted * split ratio"
