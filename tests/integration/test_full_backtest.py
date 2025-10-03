"""
Integration test: Full backtest scenarios.

Tests complete backtest workflows with real fixture data.

These tests will be fully implemented when we have:
- Strategy API (Context, on_bar, etc.)
- Backtest runner
- Output generation (NAV, fills, orders CSVs)
"""

import pytest


@pytest.mark.skip(reason="Requires Strategy API and Backtest runner (future stages)")
def test_buy_and_hold_aapl_placeholder():
    """
    Placeholder for buy-and-hold integration test.

    Future implementation:
    - Load AAPL fixture data (2020-01-02 to 2020-12-31)
    - Run buy-and-hold strategy (buy 100 shares on day 1, hold)
    - Verify final NAV
    - Compare fills.csv against golden file
    - Compare orders.csv against golden file
    """
    pass


@pytest.mark.skip(reason="Requires Strategy API and Backtest runner (future stages)")
def test_multi_symbol_rotation_placeholder():
    """
    Placeholder for multi-symbol rotation test.

    Future implementation:
    - Load AAPL, MSFT, AMZN fixture data
    - Run rotation strategy (switch between symbols based on momentum)
    - Verify position changes
    - Verify cash management across symbols
    - Test portfolio-level metrics
    """
    pass


@pytest.mark.skip(reason="Requires Strategy API and Backtest runner (future stages)")
def test_participation_with_real_volumes_placeholder():
    """
    Placeholder for participation capping with real data.

    Future implementation:
    - Load real fixture data with actual volumes
    - Submit large orders exceeding participation cap
    - Verify partial fills across multiple bars
    - Verify participation tracking per bar/side
    - Compare results against expected behavior
    """
    pass


@pytest.mark.skip(reason="Requires Strategy API and Backtest runner (future stages)")
def test_limit_orders_with_real_ohlc_placeholder():
    """
    Placeholder for limit order test with real OHLC.

    Future implementation:
    - Load real fixture bars with diverse OHLC patterns
    - Submit limit orders at various price levels
    - Verify conservative touch rules with real data
    - Test scenarios where limit just touches vs clearly crosses
    """
    pass
