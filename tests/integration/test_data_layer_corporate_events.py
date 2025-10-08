"""Integration tests for data layer - end-to-end scenarios with corporate events.

These tests validate the complete data transformation pipeline from vendor data
to canonical series, covering realistic corporate event scenarios.
"""

import datetime
from decimal import Decimal

import pytest

from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries


class TestDataLayerIntegration:
    """Integration tests for complete data layer transformation."""

    def test_aapl_2020_split_scenario(self) -> None:
        """
        Test AAPL 4:1 split on August 31, 2020 (real historical event).

        Validates:
        - Unadjusted prices match actual traded prices
        - Adjusted prices are correctly split-adjusted
        - Total return series compounds correctly through split
        - Volume adjustments are correct for all series
        """
        bars = [
            # August 28, 2020 - Before split
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 28),
                Ticker="AAPL",
                Open=498.31,
                High=500.23,
                Low=495.09,
                Close=499.23,
                MarketHoursVolume=44433000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
            # August 31, 2020 - Split effective date (4:1, inverse = 0.25)
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 8, 31),
                Ticker="AAPL",
                Open=127.58,
                High=131.00,
                Low=126.00,
                Close=129.04,
                MarketHoursVolume=210594400,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=0.25,  # Inverse of 4:1 split
                AdjustmentReason="Subdiv",
            ),
            # September 1, 2020 - After split
            AlgoseekBar(
                TradeDate=datetime.datetime(2020, 9, 1),
                Ticker="AAPL",
                Open=132.76,
                High=134.80,
                Low=130.53,
                Close=134.18,
                MarketHoursVolume=143947600,
                CumulativePriceFactor=4.0,
                CumulativeVolumeFactor=4.0,
                AdjustmentFactor=None,
                AdjustmentReason=None,
            ),
        ]

        series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # ========== UNADJUSTED SERIES ==========
        # Should match actual traded prices and volumes
        unadj = canonical["unadjusted"].bars

        # Before split
        assert unadj[0].close == 499.23
        assert unadj[0].volume == 44433000

        # Split date
        assert unadj[1].close == 129.04
        assert unadj[1].volume == 210594400

        # After split
        assert unadj[2].close == 134.18
        assert unadj[2].volume == 143947600

        # ========== ADJUSTED SERIES (Split-adjusted) ==========
        # All prices in post-split terms (÷4 for pre-split dates)
        adj = canonical["adjusted"].bars

        # Before split: adjusted to post-split terms
        assert pytest.approx(adj[0].close, abs=0.01) == 124.81  # 499.23 / 4
        assert adj[0].volume == 177732000  # 44433000 * 4

        # Split date: already in post-split terms
        assert pytest.approx(adj[1].close, abs=0.01) == 129.04
        assert adj[1].volume == 210594400

        # After split: no adjustment needed
        assert pytest.approx(adj[2].close, abs=0.01) == 134.18
        assert adj[2].volume == 143947600

        # ========== TOTAL RETURN SERIES ==========
        # Forward compounding with no look-ahead
        tr = canonical["total_return"].bars

        # First bar: TR starts at unadjusted price
        assert tr[0].close == 499.23

        # After split: TR = TR_0 * (UnAdj_1 * Split + Div) / UnAdj_0
        # TR_1 = 499.23 * (129.04 * 4 + 0) / 499.23 = 516.16
        expected_tr_1 = 499.23 * (129.04 * 4.0) / 499.23
        assert pytest.approx(tr[1].close, abs=0.1) == expected_tr_1

        # Next day: TR_2 = TR_1 * UnAdj_2 / UnAdj_1
        expected_tr_2 = expected_tr_1 * 134.18 / 129.04
        assert pytest.approx(tr[2].close, abs=0.1) == expected_tr_2

        # Volume in starting-date units (÷ cumulative split ratio)
        assert tr[0].volume == 44433000  # Pre-split (no adjustment)
        assert tr[1].volume == 52648600  # 210594400 / 4
        assert tr[2].volume == 35986900  # 143947600 / 4

    def test_dividend_with_price_gap(self) -> None:
        """
        Test typical dividend scenario where price gaps down on ex-date.

        Validates:
        - Dividend recorded on ex-date in unadjusted and adjusted series
        - Price gap reflects dividend payout
        - Total return compensates for dividend through reinvestment
        """
        bars = [
            # Day before ex-date
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 8, 10),
                Ticker="XYZ",
                Open=100.0,
                High=102.0,
                Low=99.5,
                Close=101.0,
                MarketHoursVolume=1000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # Ex-dividend date: $2.00 dividend on $101.0 close
            # Price gaps down ~$2 (not exact due to market dynamics)
            # AdjustmentFactor = (101.0 - 2.0) / 101.0 = 0.980198...
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 8, 11),
                Ticker="XYZ",
                Open=99.0,
                High=100.0,
                Low=98.5,
                Close=99.5,
                MarketHoursVolume=1200000,
                CumulativePriceFactor=1.019801980198020,  # Includes $2 div
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=0.980198019801980,  # (101.0 - 2.0) / 101.0
                AdjustmentReason="CashDiv",
            ),
            # After ex-date
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 8, 14),
                Ticker="XYZ",
                Open=99.5,
                High=101.0,
                Low=99.0,
                Close=100.5,
                MarketHoursVolume=1100000,
                CumulativePriceFactor=1.019801980198020,
                CumulativeVolumeFactor=1.0,
            ),
        ]

        series = AlgoseekPriceSeries(symbol="XYZ", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED: Dividend recorded, prices show gap
        unadj = canonical["unadjusted"].bars
        assert unadj[0].close == 101.0
        assert unadj[1].close == 99.5  # Gap down
        assert unadj[1].dividend == Decimal("2.0")
        assert unadj[2].close == 100.5

        # ADJUSTED: Same as unadjusted (no split), dividend recorded
        adj = canonical["adjusted"].bars
        assert adj[1].dividend is not None

        # TOTAL RETURN: Should compensate for dividend
        tr = canonical["total_return"].bars
        assert tr[0].close == 101.0

        # After dividend: TR = TR_0 * (UnAdj_1 + Div) / UnAdj_0
        # TR_1 = 101 * (99.5 + 2.0) / 101 = 101.5
        expected_tr_1 = 101.0 * (99.5 + 2.0) / 101.0
        assert pytest.approx(tr[1].close, abs=0.01) == expected_tr_1

        # TR should be higher than unadjusted price (dividend reinvested)
        assert tr[1].close > unadj[1].close

        # Next day: TR continues compounding
        expected_tr_2 = expected_tr_1 * 100.5 / 99.5
        assert pytest.approx(tr[2].close, abs=0.01) == expected_tr_2

    def test_reverse_split_scenario(self) -> None:
        """
        Test 1:10 reverse split (consolidation).

        Validates:
        - Prices multiply by reverse split ratio
        - Volume divides by reverse split ratio
        - All three series handle reverse split correctly
        """
        bars = [
            # Before reverse split: $2.50
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 5, 15),
                Ticker="ABC",
                Open=2.45,
                High=2.60,
                Low=2.40,
                Close=2.50,
                MarketHoursVolume=5000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # Reverse split: 1:10 (inverse: AdjustmentFactor = 10.0)
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 5, 16),
                Ticker="ABC",
                Open=24.5,
                High=26.0,
                Low=24.0,
                Close=25.0,
                MarketHoursVolume=480000,
                CumulativePriceFactor=0.1,
                CumulativeVolumeFactor=0.1,
                AdjustmentFactor=10.0,  # Inverse of 0.1 (1:10 reverse split)
                AdjustmentReason="Subdiv",
            ),
            # After reverse split: $26
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 5, 17),
                Ticker="ABC",
                Open=25.0,
                High=27.0,
                Low=24.5,
                Close=26.0,
                MarketHoursVolume=450000,
                CumulativePriceFactor=0.1,
                CumulativeVolumeFactor=0.1,
            ),
        ]

        series = AlgoseekPriceSeries(symbol="ABC", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED: Shows actual traded prices
        unadj = canonical["unadjusted"].bars
        assert unadj[0].close == 2.50  # Pre-split
        assert unadj[1].close == 25.0  # 10x higher after reverse split
        assert unadj[2].close == 26.0

        # ADJUSTED: All in post-split terms
        adj = canonical["adjusted"].bars
        assert pytest.approx(adj[0].close, abs=0.01) == 25.0  # 2.50 * 10
        assert pytest.approx(adj[1].close, abs=0.01) == 25.0
        assert pytest.approx(adj[2].close, abs=0.01) == 26.0

        # Volume adjusted
        assert adj[0].volume == 500000  # 5000000 / 10
        assert adj[1].volume == 480000
        assert adj[2].volume == 450000

        # TOTAL RETURN: Compounds through reverse split
        tr = canonical["total_return"].bars
        assert tr[0].close == 2.50

        # After reverse split: TR = TR_0 * (UnAdj_1 * Split + Div) / UnAdj_0
        # TR_1 = 2.50 * (25.0 * 0.1 + 0) / 2.50 = 2.50
        # (No gain from reverse split itself)
        expected_tr_1 = 2.50 * (25.0 * 0.1) / 2.50
        assert pytest.approx(tr[1].close, abs=0.01) == expected_tr_1

    def test_multiple_corporate_events_sequence(self) -> None:
        """
        Test complex sequence: dividend → split → dividend.

        Validates:
        - Multiple events handled correctly in sequence
        - Adjustments cascade properly
        - Total return compounds all events
        """
        bars = [
            # Initial state
            AlgoseekBar(
                TradeDate=datetime.datetime(2022, 3, 1),
                Ticker="XYZ",
                Open=198.0,
                High=202.0,
                Low=197.0,
                Close=200.0,
                MarketHoursVolume=10000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # First dividend: $1.00 on $200.0 close
            # AdjustmentFactor = (200.0 - 1.0) / 200.0 = 0.995
            AlgoseekBar(
                TradeDate=datetime.datetime(2022, 3, 15),
                Ticker="XYZ",
                Open=199.0,
                High=201.0,
                Low=198.0,
                Close=199.5,
                MarketHoursVolume=12000000,
                CumulativePriceFactor=1.005,
                CumulativeVolumeFactor=1.0,
                AdjustmentFactor=0.995,  # (200.0 - 1.0) / 200.0
                AdjustmentReason="CashDiv",
            ),
            # Before split
            AlgoseekBar(
                TradeDate=datetime.datetime(2022, 6, 1),
                Ticker="XYZ",
                Open=398.0,
                High=402.0,
                Low=395.0,
                Close=400.0,
                MarketHoursVolume=15000000,
                CumulativePriceFactor=1.005,
                CumulativeVolumeFactor=1.0,
            ),
            # 2:1 Split (inverse: AdjustmentFactor = 0.5)
            AlgoseekBar(
                TradeDate=datetime.datetime(2022, 6, 2),
                Ticker="XYZ",
                Open=199.0,
                High=201.0,
                Low=197.0,
                Close=200.0,
                MarketHoursVolume=28000000,
                CumulativePriceFactor=2.01,
                CumulativeVolumeFactor=2.0,
                AdjustmentFactor=0.5,  # Inverse of 2:1 split
                AdjustmentReason="Subdiv",
            ),
            # Second dividend: $0.50 (post-split) on $200.0 close
            # AdjustmentFactor = (200.0 - 0.5) / 200.0 = 0.9975
            AlgoseekBar(
                TradeDate=datetime.datetime(2022, 9, 15),
                Ticker="XYZ",
                Open=201.0,
                High=203.0,
                Low=200.0,
                Close=202.0,
                MarketHoursVolume=20000000,
                CumulativePriceFactor=2.015,
                CumulativeVolumeFactor=2.0,
                AdjustmentFactor=0.9975,  # (200.0 - 0.5) / 200.0
                AdjustmentReason="CashDiv",
            ),
        ]

        series = AlgoseekPriceSeries(symbol="XYZ", bars=bars)
        canonical = series.to_canonical_series()

        # UNADJUSTED: Both dividends recorded
        unadj = canonical["unadjusted"].bars
        assert unadj[1].dividend == Decimal("1.0")
        assert unadj[4].dividend == Decimal("0.5")

        # Prices show actual trading values
        assert unadj[0].close == 200.0
        assert unadj[2].close == 400.0  # Pre-split
        assert unadj[3].close == 200.0  # Post-split
        assert unadj[4].close == 202.0

        # ADJUSTED: All in post-split terms
        adj = canonical["adjusted"].bars
        # Pre-split bars adjusted
        assert pytest.approx(adj[0].close, abs=0.01) == 100.0  # 200 / 2
        assert pytest.approx(adj[2].close, abs=0.01) == 200.0  # 400 / 2
        # Post-split unchanged
        assert pytest.approx(adj[3].close, abs=0.01) == 200.0
        assert pytest.approx(adj[4].close, abs=0.01) == 202.0

        # Dividends adjusted for split
        assert adj[1].dividend == Decimal("0.5")  # 1.0 / 2 (adjusted for split)
        assert adj[4].dividend == Decimal("0.5")  # Already post-split

        # TOTAL RETURN: All events compounded
        tr = canonical["total_return"].bars
        assert tr[0].close == 200.0

        # After first dividend
        tr_1 = 200.0 * (199.5 + 1.0) / 200.0

        # Progress to split
        tr_2 = tr_1 * 400.0 / 199.5

        # After split
        tr_3 = tr_2 * (200.0 * 2.0) / 400.0

        # After second dividend
        expected_tr_4 = tr_3 * (202.0 + 0.5) / 200.0

        assert pytest.approx(tr[4].close, rel=0.01) == expected_tr_4

        # TR should be higher than simple price change due to dividends
        # Simple return: 202/200 = 1.01 (1% gain)
        # TR return: tr[4] / tr[0] should be > 1.01 due to dividends
        tr_return = tr[4].close / tr[0].close
        simple_return = unadj[4].close / unadj[0].close
        assert tr_return > simple_return

    def test_no_corporate_events_consistency(self) -> None:
        """
        Test that with no corporate events, all series remain consistent.

        Validates:
        - No adjustments → all series identical
        - Volume unchanged across series
        - No dividends recorded
        """
        bars = [
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 1, 1 + i),
                Ticker="STABLE",
                Open=100.0 + i * 0.5,
                High=102.0 + i * 0.5,
                Low=99.0 + i * 0.5,
                Close=101.0 + i * 0.5,
                MarketHoursVolume=1000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            )
            for i in range(10)
        ]

        series = AlgoseekPriceSeries(symbol="STABLE", bars=bars)
        canonical = series.to_canonical_series()

        # All three series should be identical
        for i in range(10):
            unadj_bar = canonical["unadjusted"].bars[i]
            adj_bar = canonical["adjusted"].bars[i]
            tr_bar = canonical["total_return"].bars[i]

            # Prices identical
            assert unadj_bar.open == adj_bar.open == tr_bar.open
            assert unadj_bar.high == adj_bar.high == tr_bar.high
            assert unadj_bar.low == adj_bar.low == tr_bar.low
            assert unadj_bar.close == adj_bar.close == tr_bar.close

            # Volume identical
            assert unadj_bar.volume == adj_bar.volume == tr_bar.volume

            # No dividends
            assert unadj_bar.dividend is None
            assert adj_bar.dividend is None
            assert tr_bar.dividend is None

    def test_fractional_split_ratio(self) -> None:
        """
        Test split with non-integer ratio (e.g., 3:2 split = 1.5).

        Validates:
        - Fractional split ratios handled correctly
        - Decimal precision maintained
        """
        bars = [
            # Before 3:2 split
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 3, 1),
                Ticker="XYZ",
                Open=98.0,
                High=102.0,
                Low=97.0,
                Close=100.0,
                MarketHoursVolume=1000000,
                CumulativePriceFactor=1.0,
                CumulativeVolumeFactor=1.0,
            ),
            # 3:2 split (ratio = 1.5, inverse = 0.6667)
            AlgoseekBar(
                TradeDate=datetime.datetime(2023, 3, 2),
                Ticker="XYZ",
                Open=65.33,
                High=68.0,
                Low=64.67,
                Close=66.67,
                MarketHoursVolume=1500000,
                CumulativePriceFactor=1.5,
                CumulativeVolumeFactor=1.5,
                AdjustmentFactor=0.666666666666667,  # Inverse of 1.5
                AdjustmentReason="Subdiv",
            ),
        ]

        series = AlgoseekPriceSeries(symbol="XYZ", bars=bars)
        canonical = series.to_canonical_series()

        # ADJUSTED: First bar adjusted by 1.5
        adj = canonical["adjusted"].bars
        assert pytest.approx(adj[0].close, abs=0.01) == 66.67  # 100 / 1.5
        assert adj[0].volume == 1500000  # 1M * 1.5

        # TOTAL RETURN: Compounds with 1.5 ratio
        tr = canonical["total_return"].bars
        expected_tr_1 = 100.0 * (66.67 * 1.5) / 100.0
        assert pytest.approx(tr[1].close, abs=0.1) == expected_tr_1
