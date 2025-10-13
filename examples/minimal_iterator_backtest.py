"""
Example: Minimal iterator-based backtest demonstration (Phase 4 prototype).

This example demonstrates the new iterator-based architecture without
modifying the existing backtest engine. It shows:

1. Loading data using DataLoader (returns PriceSeriesIterator)
2. Using BarMerger for multi-symbol coordination
3. Strategy receiving MultiModeBar and selecting appropriate mode
4. Execution using unadjusted prices
5. Performance tracking using total_return prices

This is a prototype for Phase 4 - demonstrating the new architecture
before updating the full backtest engine.
"""

from decimal import Decimal
from typing import Dict, List, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.data import BarMerger, DataLoader
from qtrader.models.multi_bar import MultiModeBar

logger = LoggerFactory.get_logger()


class MinimalStrategy:
    """
    Minimal strategy demonstrating MultiModeBar usage.

    Shows how strategies will work in Phase 4:
    - Receive MultiModeBar (all 3 modes)
    - Select adjusted mode for signal generation
    - Return signal decisions
    """

    def __init__(self, name: str = "MinimalDemo"):
        """Initialize strategy."""
        self.name = name
        self.bars_processed = 0
        self.signals_generated = 0

    def on_bar(self, symbol: str, bar: MultiModeBar) -> Optional[str]:
        """
        Process bar and generate signal.

        Args:
            symbol: Symbol for this bar
            bar: MultiModeBar with all 3 adjustment modes

        Returns:
            Signal string ("BUY", "SELL", "HOLD") or None
        """
        self.bars_processed += 1

        # Strategy uses ADJUSTED mode for consistent indicators across splits
        strategy_bar = bar.adjusted

        # Simple logic: buy on first bar, hold thereafter
        if self.bars_processed == 1:
            logger.info(
                "strategy.signal_generated",
                symbol=symbol,
                signal="BUY",
                trade_date=strategy_bar.trade_datetime,
                price=float(strategy_bar.close),
                mode="adjusted",
            )
            self.signals_generated += 1
            return "BUY"

        return "HOLD"


class MinimalExecutionEngine:
    """
    Minimal execution engine demonstrating mode selection.

    Shows how execution will work in Phase 4:
    - Receive MultiModeBar (all 3 modes)
    - Select unadjusted mode for realistic fills
    - Track position and cash
    """

    def __init__(self, initial_cash: Decimal):
        """Initialize execution engine."""
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}
        self.fills: List[Dict] = []

    def process_signal(self, symbol: str, signal: str, bar: MultiModeBar) -> Optional[Dict]:
        """
        Process signal and generate fill.

        Args:
            symbol: Symbol for this signal
            signal: Signal type ("BUY", "SELL", "HOLD")
            bar: MultiModeBar with all 3 modes

        Returns:
            Fill dict or None
        """
        if signal not in ("BUY", "SELL"):
            return None

        # Execution uses UNADJUSTED mode for realistic fill prices
        exec_bar = bar.unadjusted

        if signal == "BUY":
            # Calculate shares to buy (90% of cash)
            buy_value = self.cash * Decimal("0.9")
            shares = int(buy_value / Decimal(str(exec_bar.close)))

            if shares > 0:
                fill_price = exec_bar.close
                fill_value = Decimal(str(fill_price)) * shares

                # Update state
                self.cash -= fill_value
                self.positions[symbol] = self.positions.get(symbol, 0) + shares

                fill = {
                    "symbol": symbol,
                    "side": "BUY",
                    "shares": shares,
                    "price": float(fill_price),
                    "value": float(fill_value),
                    "trade_date": exec_bar.trade_datetime,
                    "mode": "unadjusted",
                }

                self.fills.append(fill)

                logger.info(
                    "execution.fill_generated",
                    **fill,
                    remaining_cash=float(self.cash),
                )

                return fill

        return None


class MinimalPortfolio:
    """
    Minimal portfolio demonstrating performance tracking.

    Shows how portfolio will work in Phase 4:
    - Receive MultiModeBar (all 3 modes)
    - Select total_return mode for accurate performance
    - Track value over time
    """

    def __init__(self, initial_cash: Decimal):
        """Initialize portfolio."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}
        self.snapshots: List[Dict] = []

    def update_position(self, fill: Dict) -> None:
        """Update position from fill."""
        symbol = fill["symbol"]
        shares = fill["shares"]
        value = Decimal(str(fill["value"]))

        if fill["side"] == "BUY":
            self.positions[symbol] = self.positions.get(symbol, 0) + shares
            self.cash -= value
        elif fill["side"] == "SELL":
            self.positions[symbol] = self.positions.get(symbol, 0) - shares
            self.cash += value

    def mark_to_market(self, symbol: str, bar: MultiModeBar) -> None:
        """
        Mark position to market using current prices.

        Args:
            symbol: Symbol to update
            bar: MultiModeBar with all 3 modes
        """
        if symbol not in self.positions or self.positions[symbol] == 0:
            return

        # Portfolio uses TOTAL_RETURN mode for accurate performance
        # (includes dividend reinvestment effect)
        perf_bar = bar.total_return

        shares = self.positions[symbol]
        market_value = shares * Decimal(str(perf_bar.close))
        total_value = self.cash + market_value

        snapshot = {
            "trade_date": perf_bar.trade_datetime,
            "symbol": symbol,
            "shares": shares,
            "price": float(perf_bar.close),
            "market_value": float(market_value),
            "cash": float(self.cash),
            "total_value": float(total_value),
            "mode": "total_return",
        }

        self.snapshots.append(snapshot)

        logger.debug(
            "portfolio.mark_to_market",
            **snapshot,
        )


def run_minimal_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_cash: Decimal = Decimal("100000"),
) -> Dict:
    """
    Run minimal iterator-based backtest demonstration.

    This demonstrates Phase 4 architecture:
    1. DataLoader creates iterators (one per symbol)
    2. BarMerger coordinates multi-symbol streams
    3. Strategy selects adjusted mode for signals
    4. Execution selects unadjusted mode for fills
    5. Portfolio selects total_return mode for performance

    Args:
        symbols: List of symbols to backtest
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        initial_cash: Initial cash balance

    Returns:
        Results dict with strategy, execution, portfolio stats

    Example:
        >>> results = run_minimal_backtest(
        ...     symbols=["AAPL"],
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31"
        ... )
        >>> print(f"Bars processed: {results['strategy']['bars_processed']}")
    """
    logger.info(
        "minimal_backtest.starting",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_cash=float(initial_cash),
    )

    # Phase 4 Architecture Step 1: Create DataLoader
    loader_config = {
        "adapter": {
            "root_path": "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample",
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": "data/equity_security_master_sample.csv",
        }
    }

    loader = DataLoader(loader_config)

    # Phase 4 Architecture Step 2: Load data for all symbols (returns iterators)
    iterators = {}
    for symbol in symbols:
        try:
            iterator = loader.load_data(symbol, start_date, end_date)
            iterators[symbol] = iterator
            logger.info("minimal_backtest.data_loaded", symbol=symbol)
        except Exception as e:
            logger.error("minimal_backtest.data_load_failed", symbol=symbol, error=str(e))
            raise

    # Phase 4 Architecture Step 3: Create BarMerger for chronological coordination
    merger = BarMerger(iterators)

    logger.info(
        "minimal_backtest.merger_initialized",
        **merger.get_stats(),
    )

    # Initialize components
    strategy = MinimalStrategy()
    execution = MinimalExecutionEngine(initial_cash)
    portfolio = MinimalPortfolio(initial_cash)

    # Phase 4 Architecture Step 4: Event loop with iterator-based flow
    bars_processed = 0

    try:
        while merger.has_next():
            # Get next bar in chronological order
            symbol, bar = merger.get_next_bar()
            bars_processed += 1

            # Log progress every 100 bars
            if bars_processed % 100 == 0:
                logger.debug(
                    "minimal_backtest.progress",
                    bars_processed=bars_processed,
                    **merger.get_stats(),
                )

            # Strategy generates signals (uses ADJUSTED mode)
            signal = strategy.on_bar(symbol, bar)

            # Execution processes signals (uses UNADJUSTED mode)
            if signal:
                fill = execution.process_signal(symbol, signal, bar)

                # Portfolio updates position from fill
                if fill:
                    portfolio.update_position(fill)

            # Portfolio marks to market (uses TOTAL_RETURN mode)
            portfolio.mark_to_market(symbol, bar)

    except StopIteration:
        logger.info("minimal_backtest.iteration_complete", bars_processed=bars_processed)

    # Collect results
    results = {
        "strategy": {
            "name": strategy.name,
            "bars_processed": strategy.bars_processed,
            "signals_generated": strategy.signals_generated,
        },
        "execution": {
            "fills": len(execution.fills),
            "positions": dict(execution.positions),
            "cash": float(execution.cash),
            "fill_details": execution.fills,
        },
        "portfolio": {
            "initial_cash": float(initial_cash),
            "final_cash": float(portfolio.cash),
            "positions": dict(portfolio.positions),
            "snapshots": len(portfolio.snapshots),
            "final_value": portfolio.snapshots[-1]["total_value"] if portfolio.snapshots else float(initial_cash),
        },
        "merger_stats": merger.get_stats(),
    }

    logger.info(
        "minimal_backtest.complete",
        bars_processed=bars_processed,
        fills=len(execution.fills),
        final_value=results["portfolio"]["final_value"],
    )

    return results


def main():
    """Run minimal backtest demonstration."""
    print("=" * 80)
    print("Phase 4 Minimal Backtest Demonstration")
    print("Iterator-Based Architecture Prototype")
    print("=" * 80)

    # Configuration
    symbols = ["AAPL"]
    start_date = "2020-01-01"
    end_date = "2020-03-31"  # 3 months for quick demo
    initial_cash = Decimal("100000")

    print("\nConfiguration:")
    print(f"  Symbols: {symbols}")
    print(f"  Date Range: {start_date} to {end_date}")
    print(f"  Initial Cash: ${initial_cash:,.2f}")

    # Run backtest
    print("\nRunning backtest...")
    try:
        results = run_minimal_backtest(symbols, start_date, end_date, initial_cash)

        # Display results
        print(f"\n{'=' * 80}")
        print("RESULTS")
        print("=" * 80)

        print("\nStrategy:")
        print(f"  Name: {results['strategy']['name']}")
        print(f"  Bars Processed: {results['strategy']['bars_processed']}")
        print(f"  Signals Generated: {results['strategy']['signals_generated']}")

        print("\nExecution:")
        print(f"  Fills: {results['execution']['fills']}")
        print(f"  Final Cash: ${results['execution']['cash']:,.2f}")
        print(f"  Positions: {results['execution']['positions']}")

        print("\nPortfolio:")
        print(f"  Initial Value: ${results['portfolio']['initial_cash']:,.2f}")
        print(f"  Final Value: ${results['portfolio']['final_value']:,.2f}")
        print(f"  P&L: ${results['portfolio']['final_value'] - results['portfolio']['initial_cash']:,.2f}")
        print(
            f"  Return: {((results['portfolio']['final_value'] / results['portfolio']['initial_cash']) - 1) * 100:.2f}%"
        )

        print("\nMerger Stats:")
        print(f"  Total Symbols: {results['merger_stats']['total_symbols']}")
        print(f"  Bars Yielded: {results['merger_stats']['total_bars_yielded']}")
        print(f"  Exhausted Symbols: {results['merger_stats']['exhausted_symbols']}")

        print(f"\n{'=' * 80}")
        print("Phase 4 Architecture Validated!")
        print("=" * 80)
        print("\nKey Points Demonstrated:")
        print("  ✓ DataLoader returns PriceSeriesIterator (streaming data)")
        print("  ✓ BarMerger coordinates multi-symbol chronological order")
        print("  ✓ Strategy uses ADJUSTED mode (consistent indicators)")
        print("  ✓ Execution uses UNADJUSTED mode (realistic fills)")
        print("  ✓ Portfolio uses TOTAL_RETURN mode (accurate performance)")
        print("  ✓ Iterator-based flow (memory efficient)")
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
