"""Unit tests for Algoseek vendor-specific bar and price series models."""

import datetime
from decimal import Decimal

import pytest

from qtrader.services.data.adapters.models.algoseek import AlgoseekBar, AlgoseekPriceSeries


class TestAlgoseekBar:
    """Test AlgoseekBar model and corporate event detection."""

    def test_create_valid_bar(self) -> None:
        """Should create valid Algoseek bar."""
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2023, 1, 15),
            Ticker="AAPL",
            Open=150.0,
            High=155.0,
            Low=149.0,
            Close=153.0,
            MarketHoursVolume=10000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
            AdjustmentFactor=None,
            AdjustmentReason=None,
        )

        assert bar.Ticker == "AAPL"
        assert bar.Open == 150.0
        assert bar.Close == 153.0
        assert bar.MarketHoursVolume == 10000000

    def test_detect_cash_dividend(self) -> None:
        """Should detect cash dividend and extract amount."""
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2023, 6, 15),
            Ticker="AAPL",
            Open=150.0,
            High=155.0,
            Low=149.0,
            Close=153.0,
            MarketHoursVolume=10000000,
            CumulativePriceFactor=1.005,
            CumulativeVolumeFactor=1.0,
            AdjustmentFactor=0.995,  # Adjustment ratio for $0.75 dividend on $150 close
            AdjustmentReason="CashDiv",
        )

        assert bar.is_dividend() is True
        assert bar.is_split() is False
        # Dividend calculation requires previous close: (1 - 0.995) × 150.0 = 0.75
        assert bar.get_dividend_amount(previous_close=150.0) == Decimal("0.75")
        assert bar.get_split_ratio() is None

    def test_detect_forward_split(self) -> None:
        """Should detect forward split (e.g., 4:1) and extract ratio."""
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2020, 8, 31),
            Ticker="AAPL",
            Open=125.0,
            High=130.0,
            Low=124.0,
            Close=129.0,
            MarketHoursVolume=200000000,
            CumulativePriceFactor=4.0,
            CumulativeVolumeFactor=4.0,
            AdjustmentFactor=0.25,  # Inverse of 4:1 split
            AdjustmentReason="Subdiv",
        )

        assert bar.is_split() is True
        assert bar.is_dividend() is False
        assert bar.get_split_ratio() == Decimal("4.0")
        assert bar.get_dividend_amount(previous_close=125.0) is None

    def test_detect_reverse_split(self) -> None:
        """Should detect reverse split (e.g., 1:5) and extract ratio."""
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2023, 3, 15),
            Ticker="XYZ",
            Open=25.0,
            High=26.0,
            Low=24.5,
            Close=25.5,
            MarketHoursVolume=500000,
            CumulativePriceFactor=0.2,
            CumulativeVolumeFactor=0.2,
            AdjustmentFactor=5.0,  # Inverse of 1:5 (0.2) reverse split
            AdjustmentReason="Subdiv",
        )

        assert bar.is_split() is True
        assert bar.is_dividend() is False
        assert bar.get_split_ratio() == Decimal("0.2")
        assert bar.get_dividend_amount(previous_close=125.0) is None

    def test_no_corporate_event(self) -> None:
        """Should handle bars with no corporate events."""
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2023, 1, 15),
            Ticker="AAPL",
            Open=150.0,
            High=155.0,
            Low=149.0,
            Close=153.0,
            MarketHoursVolume=10000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
            AdjustmentFactor=None,
            AdjustmentReason=None,
        )

        assert bar.is_dividend() is False
        assert bar.is_split() is False
        assert bar.get_dividend_amount(previous_close=150.0) is None
        assert bar.get_split_ratio() is None

    def test_parse_trade_date_from_string(self) -> None:
        """Should parse TradeDate from ISO string."""
        bar = AlgoseekBar(
            TradeDate="2023-01-15",  # type: ignore
            Ticker="AAPL",
            Open=150.0,
            High=155.0,
            Low=149.0,
            Close=153.0,
            MarketHoursVolume=10000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
        )

        assert bar.TradeDate == datetime.datetime(2023, 1, 15)

    def test_validate_ohlc(self) -> None:
        """Should validate OHLC relationships."""
        # Valid bar
        bar = AlgoseekBar(
            TradeDate=datetime.datetime(2023, 1, 15),
            Ticker="AAPL",
            Open=150.0,
            High=155.0,
            Low=149.0,
            Close=153.0,
            MarketHoursVolume=10000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
        )
        assert bar.High >= bar.Low

        # Invalid: High < Low should fail
        with pytest.raises(ValueError, match="High.*< Low"):
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 1, 15),
                Ticker="AAPL",
                Open=150.0,
                High=148.0,  # Less than low!
                Low=149.0,
                Close=153.0,
                MarketHoursVolume=10000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            )


class TestAlgoseekPriceSeries:
    """Test AlgoseekPriceSeries transformation to canonical series."""

    def test_empty_series(self) -> None:
        """Should handle empty bar list."""
        series = AlgoseekPriceSeries(symbol="AAPL", bars=[])
        canonical = series.to_canonical_series()

        assert len(canonical["unadjusted"].bars) == 0
        assert len(canonical["adjusted"].bars) == 0
        assert len(canonical["total_return"].bars) == 0

    def test_single_bar_no_corporate_events(self) -> None:
        """Should transform single bar with no adjustments."""
        bars = [
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 1, 15),
                Ticker="AAPL",
                Open=150.0,
                High=155.0,
                Low=149.0,
                Close=153.0,
                MarketHoursVolume=10000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            )
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # All three series should have same values (no adjustments)
        for mode in ["unadjusted", "adjusted", "total_return"]:
            assert len(canonical[mode].bars) == 1
            bar = canonical[mode].bars[0]
            assert bar.open == 150.0
            assert bar.high == 155.0
            assert bar.low == 149.0
            assert bar.close == 153.0
            assert bar.volume == 10000000

    def test_forward_split_adjustment(self) -> None:
        """Should correctly adjust prices for 4:1 forward split."""
        bars = [
            # Before split: $500
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 28),
                Ticker="AAPL",
                Open=498.0,
                High=500.0,
                Low=495.0,
                Close=499.0,
                MarketHoursVolume=40000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
            # Split date: 4:1 split
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 31),
                Ticker="AAPL",
                Open=125.0,
                High=131.0,
                Low=124.0,
                Close=129.0,
                MarketHoursVolume=200000000,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=0.25,  # Inverse of 4:1 split
                AdjustmentReason="Subdiv",
            ),
            # After split: ~$130
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 9, 1),
                Ticker="AAPL",
                Open=130.0,
                High=135.0,
                Low=128.0,
                Close=132.0,
                MarketHoursVolume=150000000,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED: Raw prices as traded
        unadj = canonical["unadjusted"].bars
        assert unadj[0].close == 499.0  # Pre-split
        assert unadj[1].close == 129.0  # Split date
        assert unadj[2].close == 132.0  # Post-split

        # ADJUSTED: Backward adjustment (all in post-split terms)
        adj = canonical["adjusted"].bars
        assert pytest.approx(adj[0].close, abs=0.01) == 124.75  # 499 / 4
        assert pytest.approx(adj[1].close, abs=0.01) == 129.0  # Already split-adjusted
        assert pytest.approx(adj[2].close, abs=0.01) == 132.0  # Post-split

        # Volume adjusted (multiplied by split ratio for historical bars)
        assert adj[0].volume == 160000000  # 40M * 4
        assert adj[1].volume == 200000000  # Split date
        assert adj[2].volume == 150000000  # Post-split

        # TOTAL RETURN: Forward compounding
        tr = canonical["total_return"].bars
        # First bar: TR = unadjusted
        assert tr[0].close == 499.0

        # Second bar: TR after split
        # TR_1 = TR_0 * (UnAdj_1 * Split + Div) / UnAdj_0
        # TR_1 = 499 * (129 * 4 + 0) / 499 = 516
        expected_tr_1 = 499.0 * (129.0 * 4.0) / 499.0
        assert pytest.approx(tr[1].close, abs=0.1) == expected_tr_1

        # Third bar: TR continues compounding
        # TR_2 = TR_1 * (UnAdj_2 * 1 + 0) / UnAdj_1
        expected_tr_2 = expected_tr_1 * 132.0 / 129.0
        assert pytest.approx(tr[2].close, abs=0.1) == expected_tr_2

        # Volume in starting-date units (÷ cumulative split ratio)
        assert tr[0].volume == 40000000  # Pre-split
        assert tr[1].volume == 50000000  # 200M / 4
        assert tr[2].volume == 37500000  # 150M / 4

    def test_cash_dividend_adjustment(self) -> None:
        """Should correctly handle cash dividends in all three series."""
        bars = [
            # Before dividend
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 6, 14),
                Ticker="AAPL",
                Open=180.0,
                High=182.0,
                Low=179.0,
                Close=181.0,
                MarketHoursVolume=50000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
            # Ex-dividend date: $0.50 dividend on $181.0 close
            # AdjustmentFactor = (181.0 - 0.5) / 181.0 = 0.997237569...
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 6, 15),
                Ticker="AAPL",
                Open=180.0,
                High=181.0,
                Low=179.0,
                Close=180.0,
                MarketHoursVolume=60000000,
                CumulativePriceFactor=1.0027624309392265,  # Factor includes dividend
                CumulativeVolumeFactor=1.0,  # No split
                AdjustmentFactor=0.997237569060773,  # (181 - 0.5) / 181
                AdjustmentReason="CashDiv",
            ),
            # After dividend
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 6, 16),
                Ticker="AAPL",
                Open=180.0,
                High=183.0,
                Low=179.5,
                Close=182.0,
                MarketHoursVolume=55000000,
                CumulativePriceFactor=1.0027624309392265,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED: Dividend should be recorded on ex-date
        unadj = canonical["unadjusted"].bars
        assert unadj[0].dividend is None
        assert unadj[1].dividend == Decimal("0.5")
        assert unadj[2].dividend is None

        # Prices unchanged (unadjusted)
        assert unadj[0].close == 181.0
        assert unadj[1].close == 180.0
        assert unadj[2].close == 182.0

        # ADJUSTED: Dividend recorded, prices NOT adjusted (split-adjusted only)
        adj = canonical["adjusted"].bars
        assert adj[1].dividend is not None
        # Prices should be same as unadjusted (no split)
        assert adj[0].close == 181.0
        assert adj[1].close == 180.0
        assert adj[2].close == 182.0

        # TOTAL RETURN: Dividend embedded in prices, no dividend field
        tr = canonical["total_return"].bars
        assert tr[0].dividend is None
        assert tr[1].dividend is None
        assert tr[2].dividend is None

        # TR should compound with dividend reinvestment
        # TR_1 = TR_0 * (UnAdj_1 + Div) / UnAdj_0
        # TR_1 = 181 * (180 + 0.5) / 181 = 180.5
        expected_tr_1 = 181.0 * (180.0 + 0.5) / 181.0
        assert pytest.approx(tr[1].close, abs=0.01) == expected_tr_1

        # TR_2 = TR_1 * UnAdj_2 / UnAdj_1
        expected_tr_2 = expected_tr_1 * 182.0 / 180.0
        assert pytest.approx(tr[2].close, abs=0.01) == expected_tr_2

    def test_split_and_dividend_combined(self) -> None:
        """Should correctly handle split followed by dividend."""
        bars = [
            # Pre-split
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 28),
                Ticker="AAPL",
                Open=498.0,
                High=500.0,
                Low=495.0,
                Close=499.0,
                MarketHoursVolume=40000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # 4:1 Split
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 31),
                Ticker="AAPL",
                Open=125.0,
                High=131.0,
                Low=124.0,
                Close=129.0,
                MarketHoursVolume=200000000,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=0.25,  # Inverse of 4:1 split
                AdjustmentReason="Subdiv",
            ),
            # Post-split, pre-dividend
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 11, 5),
                Ticker="AAPL",
                Open=118.0,
                High=120.0,
                Low=117.0,
                Close=119.0,
                MarketHoursVolume=100000000,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
            ),
            # Dividend: $0.20 (post-split) on $119.0 close
            # AdjustmentFactor = (119.0 - 0.2) / 119.0 = 0.998319328...
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 11, 6),
                Ticker="AAPL",
                Open=118.5,
                High=119.5,
                Low=117.5,
                Close=118.5,
                MarketHoursVolume=90000000,
                CumulativePriceFactor=4.006722689075631,  # Includes dividend
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=0.998319327731092,  # (119 - 0.2) / 119
                AdjustmentReason="CashDiv",
            ),
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED
        unadj = canonical["unadjusted"].bars
        assert unadj[0].close == 499.0  # Pre-split
        assert unadj[1].close == 129.0  # Split
        assert unadj[2].close == 119.0  # Post-split
        assert unadj[3].close == 118.5  # Dividend date
        assert unadj[3].dividend == Decimal("0.2")

        # ADJUSTED: Split-adjusted, dividend recorded
        adj = canonical["adjusted"].bars
        # Pre-split adjusted to post-split terms
        assert pytest.approx(adj[0].close, abs=0.01) == 124.75  # 499 / 4

        # Dividend adjusted for split (already post-split, so no adjustment needed)
        assert adj[3].dividend == Decimal("0.2")

        # TOTAL RETURN: Both split and dividend compounded
        tr = canonical["total_return"].bars
        # After split: TR compounds
        tr_after_split = 499.0 * (129.0 * 4.0) / 499.0  # = 516

        # After next bar: TR * (119 * 1) / 129
        tr_before_div = tr_after_split * 119.0 / 129.0

        # After dividend: TR * (118.5 + 0.2) / 119
        expected_tr_3 = tr_before_div * (118.5 + 0.2) / 119.0

        assert pytest.approx(tr[3].close, rel=0.01) == expected_tr_3

    def test_multiple_dividends(self) -> None:
        """Should correctly handle multiple dividends in sequence."""
        bars = [
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 3, 14),
                Ticker="AAPL",
                Open=150.0,
                High=152.0,
                Low=149.0,
                Close=151.0,
                MarketHoursVolume=50000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # First dividend: $0.50 on $151.0 close
            # AdjustmentFactor = (151.0 - 0.5) / 151.0 = 0.996688742...
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 3, 15),
                Ticker="AAPL",
                Open=150.0,
                High=151.0,
                Low=149.5,
                Close=150.5,
                MarketHoursVolume=60000000,
                CumulativePriceFactor=1.003311258,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=0.996688741721854,  # (151 - 0.5) / 151
                AdjustmentReason="CashDiv",
            ),
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 6, 14),
                Ticker="AAPL",
                Open=180.0,
                High=182.0,
                Low=179.0,
                Close=181.0,
                MarketHoursVolume=55000000,
                CumulativePriceFactor=1.003311258,
                CumulativeVolumeFactor=1.0,
            ),
            # Second dividend: $0.50 on $181.0 close
            # AdjustmentFactor = (181.0 - 0.5) / 181.0 = 0.997237569...
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 6, 15),
                Ticker="AAPL",
                Open=180.5,
                High=181.5,
                Low=179.5,
                Close=180.5,
                MarketHoursVolume=58000000,
                CumulativePriceFactor=1.006633880829015,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=0.997237569060773,  # (181 - 0.5) / 181
                AdjustmentReason="CashDiv",
            ),
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # Both dividends should be recorded in unadjusted and adjusted
        unadj = canonical["unadjusted"].bars
        assert unadj[1].dividend == Decimal("0.5")
        assert unadj[3].dividend == Decimal("0.5")

        # TOTAL RETURN should compound both dividends
        tr = canonical["total_return"].bars

        # After first dividend
        tr_1 = 151.0 * (150.5 + 0.5) / 151.0  # = 151

        # Continue to bar 2
        tr_2 = tr_1 * 181.0 / 150.5

        # After second dividend
        expected_tr_3 = tr_2 * (180.5 + 0.5) / 181.0

        assert pytest.approx(tr[3].close, rel=0.01) == expected_tr_3
