"""
Golden baseline test for SMA Crossover strategy.

This test ensures deterministic behavior by comparing backtest results
against a previously generated golden file.
"""

import json
from decimal import Decimal
from pathlib import Path

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.config.data_config import BarSchemaConfig, DataConfig
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy


def load_golden(strategy_name: str) -> dict:
    """Load golden baseline file."""
    golden_path = Path(__file__).parent / f"{strategy_name}_golden.json"
    with open(golden_path) as f:
        return json.load(f)


def load_strategy_and_run(golden: dict) -> dict:
    """Load strategy from examples and run backtest."""
    import importlib.util
    import sys

    # Get strategy name and find the strategy file
    strategy_name = golden["metadata"]["strategy"]
    strategy_file = Path(__file__).parent.parent.parent.parent / f"examples/{strategy_name}_strategy.py"

    # Load strategy module
    spec = importlib.util.spec_from_file_location(f"{strategy_name}_module", strategy_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Find strategy class
    strategy_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if isinstance(item, type) and hasattr(item, "on_bar") and item.__module__ == module.__name__:
            strategy_class = item
            break

    if strategy_class is None:
        raise ValueError(f"No strategy class found in {strategy_file}")

    # Get strategy config and backtest config
    strategy_config = getattr(module, "config", {})
    backtest_config = module.backtest_config

    # Load data
    resolver = DataSourceResolver()
    instruments = backtest_config["instruments"]

    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="MarketHoursVolume",
    )
    data_config = DataConfig(bar_schema=bar_schema)

    all_bars = []
    for instrument in instruments:
        adapter = resolver.resolve(instrument)
        bars = list(adapter.read_bars(data_config))
        all_bars.extend(bars)

    all_bars.sort(key=lambda b: (b.ts, b.symbol))

    # Create portfolio and risk manager
    portfolio = Portfolio(initial_cash=Decimal(str(backtest_config["initial_cash"])))

    risk_policy = RiskPolicy(
        default_position_size=Decimal(str(backtest_config["position_size"])),
        max_position_pct=Decimal(str(backtest_config["max_position_pct"])),
        allow_shorting=backtest_config["allow_shorting"],
    )

    risk_manager = RiskManager(portfolio=portfolio, policy=risk_policy)

    ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

    # Create execution config
    exec_config = ExecutionConfig(
        warmup=backtest_config.get("warmup", False),
        warmup_bars=backtest_config.get("warmup_bars"),
        max_participation=Decimal(str(backtest_config.get("max_participation", 0.10))),
    )

    # Instantiate strategy with config
    if strategy_config:
        strategy = strategy_class(**strategy_config)
    else:
        strategy = strategy_class()

    backtest_obj = Backtest(config=exec_config, strategy=strategy)

    symbols = [inst.symbol for inst in instruments]
    out_dir = Path("backtest_results") / "test_golden"
    out_dir.mkdir(parents=True, exist_ok=True)

    _ = backtest_obj.run(ctx=ctx, bars=all_bars, symbols=symbols, out_dir=out_dir)

    # Extract results
    results = {
        "final_cash": float(portfolio.cash.get_balance()),
        "final_equity": float(portfolio.get_equity()),
        "num_trades": len([fill for fill in backtest_obj.all_fills if fill.qty != 0]),
        "num_fills": len(backtest_obj.all_fills),
        "total_commissions": float(sum(fill.fees for fill in backtest_obj.all_fills)),
        "final_positions": {
            symbol: {"quantity": float(pos.qty), "avg_price": float(pos.avg_price)}
            for symbol, pos in portfolio.positions.get_all_positions().items()
            if not pos.is_flat()
        },
    }

    return results


def assert_close(actual: float, expected: float, rel_tol: float = 1e-6, abs_tol: float = 0.01):
    """Assert two floats are close within tolerance."""
    if abs(expected) < abs_tol:
        # For values close to zero, use absolute tolerance
        assert abs(actual - expected) < abs_tol, f"Expected {expected}, got {actual}"
    else:
        # For larger values, use relative tolerance
        rel_diff = abs((actual - expected) / expected)
        assert rel_diff < rel_tol, f"Expected {expected}, got {actual} (rel_diff={rel_diff:.2%})"


def test_sma_crossover_matches_golden():
    """Verify SMA crossover results match golden file exactly."""
    golden = load_golden("sma_crossover")
    results = load_strategy_and_run(golden)

    # Compare key metrics
    assert_close(results["final_cash"], golden["results"]["final_cash"])
    assert_close(results["final_equity"], golden["results"]["final_equity"])
    assert_close(results["total_commissions"], golden["results"]["total_commissions"])

    # Compare trade counts (should be exact)
    assert results["num_trades"] == golden["results"]["num_trades"], (
        f"Trade count mismatch: expected {golden['results']['num_trades']}, got {results['num_trades']}"
    )
    assert results["num_fills"] == golden["results"]["num_fills"], (
        f"Fill count mismatch: expected {golden['results']['num_fills']}, got {results['num_fills']}"
    )

    # Compare final positions
    assert len(results["final_positions"]) == len(golden["final_positions"]), (
        f"Position count mismatch: expected {len(golden['final_positions'])}, got {len(results['final_positions'])}"
    )

    for symbol, position in results["final_positions"].items():
        assert symbol in golden["final_positions"], f"Symbol {symbol} not in golden positions"
        golden_pos = golden["final_positions"][symbol]

        assert_close(position["quantity"], golden_pos["quantity"])
        assert_close(position["avg_price"], golden_pos["avg_price"])


if __name__ == "__main__":
    # Allow running test directly for debugging
    test_sma_crossover_matches_golden()
    print("✅ SMA crossover golden test passed!")
