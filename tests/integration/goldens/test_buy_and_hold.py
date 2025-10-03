"""
Golden file regression tests: Buy-and-Hold strategy.

Verifies deterministic output by comparing backtest results against
golden files (reference outputs).

These tests ensure:
- Deterministic execution (same inputs → same outputs)
- No regressions in calculations (NAV, PnL, fills)
- Stable API (output format doesn't change unexpectedly)
"""

import pytest


@pytest.mark.skip(reason="Requires Strategy API, Backtest runner, and golden file generation")
def test_buy_and_hold_aapl_golden_placeholder():
    """
    Placeholder for buy-and-hold AAPL golden test.

    Future implementation:
    - Run buy-and-hold on AAPL (fixture data)
    - Compare nav.csv against golden file
    - Compare fills.csv against golden file
    - Compare orders.csv against golden file
    - Compare run.json metadata against golden

    Golden files stored in: tests/integration/goldens/data/
    """
    pass


@pytest.mark.skip(reason="Requires Strategy API, Backtest runner, and golden file generation")
def test_buy_and_hold_msft_golden_placeholder():
    """
    Placeholder for buy-and-hold MSFT golden test.
    """
    pass


@pytest.mark.skip(reason="Requires Strategy API, Backtest runner, and golden file generation")
def test_buy_and_hold_amzn_golden_placeholder():
    """
    Placeholder for buy-and-hold AMZN golden test.
    """
    pass
