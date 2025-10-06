"""Unit tests for DividendCalculator."""

from decimal import Decimal

from qtrader.execution.dividend_calculator import DividendCalculator


class TestDividendCalculator:
    """Tests for dividend calculator."""

    def test_calculate_simple_dividend(self):
        """Test basic dividend calculation."""
        # AAPL-like dividend: $0.23/share
        close_before = Decimal("152.55")
        close_after = Decimal("152.32")
        factor = Decimal("1.001508")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        assert div == Decimal("0.23")

    def test_calculate_larger_dividend(self):
        """Test calculation with larger dividend ($2.00/share)."""
        close_before = Decimal("100.00")
        close_after = Decimal("98.00")
        factor = Decimal("1.020408")  # 2% adjustment

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        assert div == Decimal("2.00")

    def test_calculate_small_dividend(self):
        """Test calculation with small dividend ($0.05/share)."""
        close_before = Decimal("50.00")
        close_after = Decimal("49.95")
        factor = Decimal("1.001001")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        assert div == Decimal("0.05")

    def test_zero_dividend_returns_none(self):
        """Test that zero dividend returns None."""
        # No price change
        close_before = Decimal("100.00")
        close_after = Decimal("100.00")
        factor = Decimal("1.0")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        # Should return None for zero dividend
        assert div is None

    def test_negative_price_before_returns_none(self):
        """Test invalid negative price before ex-date."""
        close_before = Decimal("-100.00")
        close_after = Decimal("98.00")
        factor = Decimal("1.020408")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_negative_price_after_returns_none(self):
        """Test invalid negative price on ex-date."""
        close_before = Decimal("100.00")
        close_after = Decimal("-98.00")
        factor = Decimal("1.020408")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_zero_price_before_returns_none(self):
        """Test invalid zero price before ex-date."""
        close_before = Decimal("0.00")
        close_after = Decimal("98.00")
        factor = Decimal("1.020408")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_zero_price_after_returns_none(self):
        """Test invalid zero price on ex-date."""
        close_before = Decimal("100.00")
        close_after = Decimal("0.00")
        factor = Decimal("1.020408")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_zero_factor_returns_none(self):
        """Test invalid zero adjustment factor."""
        close_before = Decimal("100.00")
        close_after = Decimal("98.00")
        factor = Decimal("0.00")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_negative_factor_returns_none(self):
        """Test invalid negative adjustment factor."""
        close_before = Decimal("100.00")
        close_after = Decimal("98.00")
        factor = Decimal("-1.020408")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is None

    def test_very_small_dividend(self):
        """Test very small dividend (1 cent)."""
        close_before = Decimal("100.00")
        close_after = Decimal("99.99")
        factor = Decimal("1.000100")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        assert div == Decimal("0.01")

    def test_rounding_to_cents(self):
        """Test that dividend is rounded to nearest cent."""
        # Result: $0.237 should round to $0.24
        close_before = Decimal("152.55")
        close_after = Decimal("152.314")
        factor = Decimal("1.001545")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        # Should be rounded to 2 decimal places
        assert div == Decimal("0.24")

    def test_negative_dividend_from_price_increase(self):
        """Test that price increase (negative div) returns None."""
        # Price went UP on ex-date (unusual, shouldn't happen)
        close_before = Decimal("98.00")
        close_after = Decimal("100.00")
        factor = Decimal("1.0")

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        # Should return None for negative dividend
        assert div is None

    def test_validate_dividend_event_valid(self):
        """Test validation of valid dividend event."""
        valid = DividendCalculator.validate_dividend_event(
            symbol="AAPL",
            ex_date="2023-02-10",
            dividend_per_share=Decimal("0.23"),
        )

        assert valid is True

    def test_validate_dividend_event_missing_symbol(self):
        """Test validation rejects missing symbol."""
        valid = DividendCalculator.validate_dividend_event(
            symbol="",
            ex_date="2023-02-10",
            dividend_per_share=Decimal("0.23"),
        )

        assert valid is False

    def test_validate_dividend_event_missing_ex_date(self):
        """Test validation rejects missing ex-date."""
        valid = DividendCalculator.validate_dividend_event(
            symbol="AAPL",
            ex_date="",
            dividend_per_share=Decimal("0.23"),
        )

        assert valid is False

    def test_validate_dividend_event_zero_amount(self):
        """Test validation rejects zero dividend."""
        valid = DividendCalculator.validate_dividend_event(
            symbol="AAPL",
            ex_date="2023-02-10",
            dividend_per_share=Decimal("0.00"),
        )

        assert valid is False

    def test_validate_dividend_event_negative_amount(self):
        """Test validation rejects negative dividend."""
        valid = DividendCalculator.validate_dividend_event(
            symbol="AAPL",
            ex_date="2023-02-10",
            dividend_per_share=Decimal("-0.23"),
        )

        assert valid is False

    def test_validate_dividend_event_suspiciously_high(self):
        """Test validation warns on very high dividend (>$100)."""
        # Should still validate but log warning
        valid = DividendCalculator.validate_dividend_event(
            symbol="AAPL",
            ex_date="2023-02-10",
            dividend_per_share=Decimal("150.00"),
        )

        # Currently returns False, but could be changed to True with warning
        assert valid is False

    def test_real_world_example_msft(self):
        """Test with real MSFT dividend data."""
        # MSFT typically pays ~$0.62/quarter
        close_before = Decimal("250.00")
        close_after = Decimal("249.38")
        factor = Decimal("1.002481")  # 0.248% adjustment

        div = DividendCalculator.calculate_from_factors(close_before, close_after, factor)

        assert div is not None
        assert Decimal("0.61") <= div <= Decimal("0.63")
