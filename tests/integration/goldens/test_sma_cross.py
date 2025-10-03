"""
Golden file regression tests: SMA Crossover strategy.

Tests a simple moving average crossover strategy with deterministic output.
"""

import pytest


@pytest.mark.skip(reason="Requires Strategy API, Backtest runner, and golden file generation")
def test_sma_cross_msft_golden_placeholder():
    """
    Placeholder for SMA crossover MSFT golden test.

    Future implementation:
    - Implement simple SMA(10) x SMA(50) crossover strategy
    - Run on MSFT fixture data
    - Verify buy/sell signals generated at correct times
    - Compare outputs against golden files
    """
    pass
