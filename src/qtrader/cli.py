"""Command-line interface."""

import importlib
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import structlog

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.config.system_config import get_config
from qtrader.execution.config import ExecutionConfig
from qtrader.models.instrument import Instrument
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """QTrader - Quantitative Trading Environment"""
    pass


@main.command()
@click.option(
    "--strategy",
    type=click.Path(exists=True),
    required=True,
    help="Path to self-contained strategy Python file",
)
@click.option(
    "--out",
    type=click.Path(),
    default=None,
    help="Output directory for results (default: ./backtest_results/<strategy_name>_<timestamp>)",
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override strategy/backtest config: --set param=value (e.g., --set warmup=true)",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    help="Enable debug output (CSV exports, detailed logs)",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    help="Verbose logging output",
)
def backtest(
    strategy: str,
    out: Optional[str],
    overrides: tuple,
    debug: bool,
    verbose: bool,
):
    """
    Run a backtest with a self-contained strategy file.

    Strategy file must contain:
    - A Strategy class implementing on_bar()
    - Optional 'config' dict with strategy parameters
    - 'backtest_config' dict with ALL backtest settings:
      - instruments: List of Instrument objects specifying symbols and data sources
      - initial_cash: Starting portfolio cash
      - position_size: Default position size
      - max_position_pct: Max position as % of portfolio
      - allow_shorting: Enable short selling
      - max_participation: Max volume participation
      - warmup: Enable indicator warmup
      - warmup_bars: Explicit warmup bars (or None for auto-detect)

    The CLI will auto-discover the Strategy class and load all configurations from the file.

    Configuration precedence: CLI --set > strategy file config > defaults

    All configuration can be overridden via --set:
    - Strategy parameters: --set fast_period=10 --set slow_period=30
    - Backtest settings: --set warmup=true --set initial_cash=200000

    Examples:

        # Basic usage (all config from strategy file)
        qtrader backtest --strategy strategies/sma_crossover.py

        # With custom output directory
        qtrader backtest --strategy strategies/sma_crossover.py --out results/exp1

        # With parameter overrides
        qtrader backtest \\
          --strategy strategies/sma_crossover.py \\
          --set fast_period=10 --set slow_period=30

        # Override data and symbols
        qtrader backtest \\
          --strategy strategies/sma_crossover.py \\
          --set 'data_paths=["data/TSLA.parquet"]' \\
          --set 'symbols=["TSLA"]'

        # Override backtest config
        qtrader backtest \\
          --strategy strategies/sma_crossover.py \\
          --set initial_cash=200000 --set warmup=true --set warmup_bars=50

        # With debug output
        qtrader backtest \\
          --strategy strategies/sma_crossover.py \\
          --debug --verbose
    """
    # Display header
    click.echo("=" * 60)
    click.echo("QTrader - Backtesting Engine")
    click.echo("=" * 60)

    strategy_path = Path(strategy)
    click.echo(f"Strategy file: {strategy_path.name}")

    # Load strategy module and extract configuration
    try:
        strategy_module = _load_strategy_module(strategy)
        strategy_class = _find_strategy_class(strategy_module)
        click.echo(f"✓ Strategy loaded: {strategy_class.__name__}")

        # Extract strategy configuration
        strategy_config = _extract_strategy_config(strategy_module, strategy_class)

        # Extract backtest configuration
        backtest_config = _extract_backtest_config(strategy_module)

    except Exception as e:
        click.echo(f"✗ Failed to load strategy: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Validate required backtest config
    if "instruments" not in backtest_config or not backtest_config["instruments"]:
        click.echo("✗ Error: 'backtest_config' must include 'instruments' (list of Instrument objects)", err=True)
        sys.exit(1)

    # Extract instruments
    instruments = backtest_config["instruments"]
    if not isinstance(instruments, list):
        click.echo("✗ Error: 'instruments' must be a list of Instrument objects", err=True)
        sys.exit(1)

    if not all(isinstance(i, Instrument) for i in instruments):
        click.echo("✗ Error: All items in 'instruments' must be Instrument objects", err=True)
        sys.exit(1)

    # Apply CLI overrides to strategy config and backtest config
    if overrides:
        strategy_config = _apply_config_overrides(strategy_config, overrides)
        # Also check if overrides contain backtest config keys
        for override in overrides:
            if "=" in override:
                key = override.split("=", 1)[0].strip()
                if key in backtest_config and key != "instruments":  # Can't override instruments via CLI
                    # Parse and apply to backtest_config
                    value_str = override.split("=", 1)[1].strip()
                    import ast

                    try:
                        value = ast.literal_eval(value_str)
                    except (ValueError, SyntaxError):
                        value = value_str

                    # Convert to Decimal for numeric backtest configs
                    if key in ["initial_cash", "position_size", "max_position_pct", "max_participation"]:
                        if not isinstance(value, Decimal):
                            backtest_config[key] = Decimal(str(value))
                        else:
                            backtest_config[key] = value
                    else:
                        backtest_config[key] = value

        click.echo(f"✓ Applied {len(overrides)} config override(s)")

    # Display configuration
    click.echo("\nStrategy Configuration:")
    for key, value in strategy_config.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nBacktest Configuration:")
    click.echo(f"  Instruments: {len(instruments)}")
    for inst in instruments:
        freq_str = f"@{inst.frequency}" if inst.frequency else ""
        click.echo(f"    - {inst.symbol}{freq_str} ({inst.instrument_type.value}, {inst.data_source.value})")
    click.echo(f"  Initial Cash: ${backtest_config['initial_cash']:,.2f}")
    click.echo(f"  Position Size: ${backtest_config['position_size']:,.2f}")
    click.echo(f"  Max Position %%: {float(backtest_config['max_position_pct']) * 100:.1f}%")
    click.echo(f"  Allow Short: {backtest_config['allow_shorting']}")

    warmup_enabled = backtest_config.get("warmup", False)
    click.echo(f"  Warmup: {'Enabled' if warmup_enabled else 'Disabled'}")
    if warmup_enabled and backtest_config.get("warmup_bars"):
        click.echo(f"  Warmup Bars: {backtest_config['warmup_bars']}")

    # Load data using DataSourceResolver
    try:
        bars = _load_data_from_instruments(instruments, verbose)
        click.echo(f"\n✓ Data loaded: {len(bars)} bars")

        # Load adjustment events (dividends, splits) for each instrument
        adjustment_events = _load_adjustments_from_instruments(instruments, verbose)
        if adjustment_events:
            total_adjustments = sum(len(events) for events in adjustment_events.values())
            click.echo(f"✓ Adjustments loaded: {total_adjustments} events")
        else:
            click.echo("ℹ No adjustment events loaded")
            adjustment_events = None
    except Exception as e:
        click.echo(f"✗ Failed to load data: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Extract symbol list from instruments
    symbol_list = [inst.symbol for inst in instruments]

    # Load system configuration
    sys_config = get_config()

    # Create output directory (default if not specified)
    if out is None:
        # Use system config for default results directory and format
        if sys_config.output.use_timestamps:
            timestamp = datetime.now().strftime(sys_config.output.timestamp_format)
            strategy_name = strategy_path.stem

            if sys_config.output.organize_by_date:
                # Format: results_dir/YYYY-MM-DD/strategy_name_timestamp/
                date_dir = datetime.now().strftime("%Y-%m-%d")
                out = f"{sys_config.output.default_results_dir}/{date_dir}/{strategy_name}_{timestamp}"
            else:
                # Format: results_dir/strategy_name_timestamp/
                out = f"{sys_config.output.default_results_dir}/{strategy_name}_{timestamp}"
        else:
            # No timestamp
            out = f"{sys_config.output.default_results_dir}/{strategy_path.stem}"

    output_path = Path(out)
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"✓ Output directory: {output_path}")

    # Initialize components with configuration
    portfolio = Portfolio(initial_cash=backtest_config["initial_cash"])

    risk_policy = RiskPolicy(
        default_position_size=backtest_config["position_size"],
        max_position_pct=backtest_config["max_position_pct"],
        allow_shorting=backtest_config["allow_shorting"],
    )
    risk_manager = RiskManager(portfolio=portfolio, policy=risk_policy)

    ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

    # Configure execution
    max_fill_price_dev = backtest_config.get("max_fill_price_deviation_pct")
    if max_fill_price_dev is not None and not isinstance(max_fill_price_dev, Decimal):
        max_fill_price_dev = Decimal(str(max_fill_price_dev))
    elif max_fill_price_dev is None:
        max_fill_price_dev = Decimal("0.10")  # Default 10%

    exec_config = ExecutionConfig(
        warmup=backtest_config.get("warmup", False),
        warmup_bars=backtest_config.get("warmup_bars"),
        max_participation=backtest_config.get("max_participation", Decimal("0.10")),
        allow_high_participation=False,
        max_fill_price_deviation_pct=max_fill_price_dev,
    )

    # Initialize strategy with configuration
    try:
        if strategy_config:
            # Try to pass config to strategy constructor
            strategy_instance = strategy_class(**strategy_config)
        else:
            strategy_instance = strategy_class()
    except TypeError:
        # Fallback: strategy doesn't accept config in __init__
        strategy_instance = strategy_class()
        # Try to set config attribute if it exists
        if hasattr(strategy_instance, "config") and strategy_config:
            for key, value in strategy_config.items():
                setattr(strategy_instance.config, key, value)

    # Run backtest
    click.echo("\n" + "=" * 60)
    click.echo("Running backtest...")
    click.echo("=" * 60 + "\n")

    backtest_obj = Backtest(exec_config, strategy_instance)

    try:
        start_time = datetime.now()
        metadata = backtest_obj.run(
            ctx=ctx,
            bars=bars,
            symbols=symbol_list,
            out_dir=output_path,
            adjustment_events=adjustment_events,
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        click.echo("\n" + "=" * 60)
        click.echo("✓ Backtest Complete")
        click.echo("=" * 60)
        click.echo(f"Duration: {duration:.2f}s")
        click.echo(f"Bars Processed: {metadata.get('trading_bars', metadata.get('total_bars', 0))}")
        click.echo(f"Total Fills: {metadata['total_fills']}")
        click.echo("\nFinal Portfolio:")
        click.echo(f"  Cash: ${metadata['final_cash']:,.2f}")
        click.echo(f"  Equity: ${metadata['final_equity']:,.2f}")

        total_value = float(metadata["final_cash"]) + float(metadata["final_equity"])
        click.echo(f"  Total Value: ${total_value:,.2f}")

        initial_cash = float(backtest_config["initial_cash"])
        pnl = total_value - initial_cash
        pnl_pct = (pnl / initial_cash) * 100
        click.echo(f"\nP&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        # Export files based on system config or debug flag
        if debug or any(sys_config.output.generate_files.values()):
            _export_output_files(backtest_obj, output_path, metadata, backtest_config, sys_config)
            click.echo(f"\n✓ Output files saved to {output_path}")
        else:
            click.echo(f"\nResults saved to: {output_path}")

    except Exception as e:
        click.echo(f"\n✗ Backtest failed: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option("--data", type=click.Path(exists=True), required=True, help="Path to data file")
@click.option("--symbols", type=str, help="Comma-separated list of symbols to check (optional)")
def validate_data(data: str, symbols: Optional[str]):
    """
    [DEPRECATED] Validate dataset without running backtest.

    This command is deprecated. Please use the new Instrument-based pattern instead.
    See docs/stage6c_instrument_abstraction.md for migration guide.
    """
    click.echo("✗ This command is deprecated.", err=True)
    click.echo("\nThe validate-data command has been replaced with the Instrument-based pattern.")
    click.echo("Please update your strategy to use Instrument objects instead of file paths.")
    click.echo("\nExample:")
    click.echo("  from qtrader.models.instrument import Instrument, InstrumentType, DataSource")
    click.echo('  instruments = [Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)]')
    click.echo("\nSee docs/stage6c_instrument_abstraction.md for details.")
    sys.exit(1)


@main.command()
def version():
    """Display QTrader version."""
    click.echo("QTrader version 0.1.0")


def _load_strategy_module(strategy_path: str):
    """
    Load a strategy module from a file path.

    Args:
        strategy_path: Path to Python file containing strategy

    Returns:
        Loaded module object

    Raises:
        ImportError: If module cannot be loaded
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["user_strategy"] = module
    spec.loader.exec_module(module)

    return module


def _find_strategy_class(module):
    """
    Find the Strategy class in a module.

    Args:
        module: Module object to search

    Returns:
        Strategy class

    Raises:
        ValueError: If no Strategy class found or multiple found
    """
    import inspect

    strategy_classes = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if class has on_bar method (required for strategies)
        if hasattr(obj, "on_bar") and obj.__module__ == module.__name__:
            strategy_classes.append((name, obj))

    if not strategy_classes:
        raise ValueError(f"No Strategy class found in {module.__name__}. Strategy must implement on_bar() method.")

    if len(strategy_classes) > 1:
        class_names = [name for name, _ in strategy_classes]
        raise ValueError(
            f"Multiple Strategy classes found: {', '.join(class_names)}. Only one Strategy class per file is supported."
        )

    return strategy_classes[0][1]


def _extract_strategy_config(module, strategy_class) -> dict:
    """
    Extract strategy configuration from module or class.

    Looks for:
    1. module-level 'config' dict or object
    2. strategy_class.config attribute
    3. Constructor signature of strategy_class

    Args:
        module: Strategy module
        strategy_class: Strategy class

    Returns:
        Dict of configuration parameters with defaults
    """
    import inspect

    config = {}

    # Try module-level config
    if hasattr(module, "config"):
        mod_config = module.config
        if isinstance(mod_config, dict):
            config.update(mod_config)
        elif hasattr(mod_config, "__dict__"):
            # Pydantic model or similar
            config.update({k: v for k, v in mod_config.__dict__.items() if not k.startswith("_")})

    # Try class-level config
    if hasattr(strategy_class, "config"):
        cls_config = strategy_class.config
        if isinstance(cls_config, dict):
            config.update(cls_config)
        elif hasattr(cls_config, "__dict__"):
            config.update({k: v for k, v in cls_config.__dict__.items() if not k.startswith("_")})

    # Extract from constructor signature
    sig = inspect.signature(strategy_class.__init__)
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.default != inspect.Parameter.empty:
            # Only add if not already present
            if param_name not in config:
                config[param_name] = param.default

    return config


def _extract_backtest_config(module) -> dict:
    """
    Extract backtest configuration from module.

    Looks for module-level 'backtest_config' dict with:
    - instruments (required): List of Instrument objects
    - initial_cash
    - position_size
    - max_position_pct
    - allow_shorting
    - max_participation
    - warmup
    - warmup_bars

    Args:
        module: Strategy module

    Returns:
        Dict of backtest configuration with defaults
    """
    defaults = {
        "instruments": [],  # Required - must be set in strategy file
        "initial_cash": Decimal("100000"),
        "position_size": Decimal("5000"),
        "max_position_pct": Decimal("0.10"),
        "allow_shorting": False,
        "max_participation": Decimal("0.10"),
        "warmup": False,
        "warmup_bars": None,
    }

    if hasattr(module, "backtest_config"):
        mod_config = module.backtest_config
        if isinstance(mod_config, dict):
            # Convert numeric values to Decimal
            for key, value in mod_config.items():
                if key in ["initial_cash", "position_size", "max_position_pct", "max_participation"]:
                    if not isinstance(value, Decimal):
                        defaults[key] = Decimal(str(value))
                    else:
                        defaults[key] = value
                else:
                    defaults[key] = value

    return defaults


def _apply_config_overrides(config: dict, overrides: tuple) -> dict:
    """
    Apply CLI overrides to configuration.

    Args:
        config: Base configuration dict
        overrides: Tuple of "key=value" strings from CLI

    Returns:
        Updated configuration dict

    Raises:
        ValueError: If override format is invalid
    """
    import ast

    config = config.copy()

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected 'key=value'")

        key, value_str = override.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Try to parse value as Python literal
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Treat as string
            value = value_str

        config[key] = value

    return config


def _load_strategy_class(strategy_path: str):
    """
    DEPRECATED: Use _load_strategy_module + _find_strategy_class instead.

    Load a strategy class from a module path.

    Args:
        strategy_path: Dotted path like 'examples.sma_crossover_strategy.SMACrossover'

    Returns:
        Strategy class

    Raises:
        ImportError: If module or class cannot be loaded
        AttributeError: If class doesn't exist in module
    """
    parts = strategy_path.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid strategy path: {strategy_path}. Expected format: 'module.submodule.ClassName'")

    module_path = ".".join(parts[:-1])
    class_name = parts[-1]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")

    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_path}' has no class '{class_name}'")

    # Verify it looks like a strategy
    if not hasattr(strategy_class, "on_bar"):
        raise TypeError(f"Class '{class_name}' does not implement on_bar() method")

    return strategy_class


def _load_data_from_instruments(instruments: List[Instrument], verbose: bool = False):
    """
    Load data from instruments using DataSourceResolver.

    Args:
        instruments: List of Instrument objects
        verbose: Enable verbose logging

    Returns:
        List of Bar objects sorted by timestamp and symbol
    """
    from qtrader.config.data_config import BarSchemaConfig, DataConfig

    bars = []
    resolver = DataSourceResolver()

    # Create default config for Algoseek data (adapter-specific)
    bar_schema = BarSchemaConfig(
        ts="TradeDate", symbol="Ticker", open="Open", high="High", low="Low", close="Close", volume="MarketHoursVolume"
    )
    config = DataConfig(bar_schema=bar_schema)

    for instrument in instruments:
        if verbose:
            logger.info("loading_data", instrument=str(instrument))

        # Resolve instrument to adapter
        adapter = resolver.resolve(instrument)

        # Load bars from adapter
        instrument_bars = list(adapter.read_bars(config))
        bars.extend(instrument_bars)

    # Sort by timestamp, then symbol (deterministic)
    bars.sort(key=lambda b: (b.ts, b.symbol))

    return bars


def _load_adjustments_from_instruments(instruments: list, verbose: bool = False):
    """
    Load adjustment events (dividends, splits) from instruments using DataSourceResolver.

    Args:
        instruments: List of Instrument objects
        verbose: Enable verbose logging

    Returns:
        Dict mapping symbol -> list of AdjustmentEvent objects, or None if no adjustments
    """
    from qtrader.config.data_config import AdjustmentSchemaConfig, BarSchemaConfig, DataConfig

    all_adjustments: Dict[str, List[Any]] = {}
    resolver = DataSourceResolver()

    # Create adjustment schema config for Algoseek data
    adj_schema = AdjustmentSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        event_type="AdjustmentReason",
        px_factor="CumulativePriceFactor",
        vol_factor="CumulativeVolumeFactor",
        metadata_fields=["AdjustmentFactor"],  # Capture individual adjustment factor
    )
    # Need bar_schema even though we're only reading adjustments
    bar_schema = BarSchemaConfig(
        ts="TradeDate", symbol="Ticker", open="Open", high="High", low="Low", close="Close", volume="MarketHoursVolume"
    )
    config = DataConfig(bar_schema=bar_schema, adjustment_schema=adj_schema)

    for instrument in instruments:
        if verbose:
            logger.info("loading_adjustments", instrument=str(instrument))

        # Resolve instrument to adapter
        adapter = resolver.resolve(instrument)

        # Load adjustments from adapter
        try:
            adjustments = list(adapter.read_adjustments(config))
            if adjustments:
                # Index by symbol
                symbol = instrument.symbol
                if symbol not in all_adjustments:
                    all_adjustments[symbol] = []
                all_adjustments[symbol].extend(adjustments)
        except (AttributeError, NotImplementedError):
            # Adapter doesn't support adjustments
            if verbose:
                logger.debug("adapter_no_adjustments", adapter=type(adapter).__name__)
            continue

    # Return None if no adjustments found
    return all_adjustments if all_adjustments else None


def _load_data_files(data_paths: tuple, symbols: list, verbose: bool = False):
    """
    DEPRECATED: Legacy function for loading data from file paths.
    Use _load_data_from_instruments() instead.

    This function is no longer supported due to the Instrument-based architecture.

    Args:
        data_paths: Tuple of file paths
        symbols: List of symbols to filter
        verbose: Enable verbose logging

    Raises:
        NotImplementedError: Always raises (deprecated)
    """
    raise NotImplementedError(
        "Legacy data loading not supported. Please use Instrument objects in backtest_config instead of data_paths."
    )


def _export_output_files(
    backtest_obj: Backtest,
    output_dir: Path,
    metadata: Dict[str, Any],
    backtest_config: Dict[str, Any],
    sys_config,
):
    """
    Export backtest output files based on system configuration.

    Args:
        backtest_obj: Backtest instance with results
        output_dir: Directory to write files
        metadata: Backtest metadata dict
        backtest_config: Strategy backtest configuration
        sys_config: System configuration with generate_files settings
    """
    import csv
    import json

    generate = sys_config.output.generate_files

    # 1. Metadata file (JSON)
    if generate.get("metadata", True):
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("exported_metadata", path=str(metadata_file))

    # 2. Fills file (CSV)
    if generate.get("fills", True) and backtest_obj.all_fills:
        fills_file = output_dir / "fills.csv"
        with open(fills_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fill_id",
                    "order_id",
                    "timestamp",
                    "symbol",
                    "side",
                    "qty",
                    "price",
                    "fees",
                    "slippage_bps",
                ]
            )

            for fill in backtest_obj.all_fills:
                writer.writerow(
                    [
                        fill.fill_id,
                        fill.order_id,
                        fill.execution_ts,
                        fill.symbol,
                        fill.side.name,
                        fill.qty,
                        fill.price,
                        fill.fees,
                        fill.slippage_bps,
                    ]
                )

        logger.info("exported_fills", path=str(fills_file), count=len(backtest_obj.all_fills))

    # 3. Portfolio snapshots file (CSV)
    if generate.get("portfolio", True) and backtest_obj.portfolio_snapshots:
        snapshots_file = output_dir / "portfolio.csv"
        with open(snapshots_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    # Bar OHLC data
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    # Adjustment factors
                    "cumulative_price_factor",
                    "cumulative_volume_factor",
                    "adjustment_factor",
                    "adjustment_reason",
                    # Fill information
                    "signal",
                    "order_id",
                    "order_type",
                    "fill_qty",
                    "fill_price",
                    "commission",
                    # Cash flow tracking
                    "initial_cash",
                    "cash_debits",
                    "cash_credits",
                    "end_cash",
                    # Portfolio tracking
                    "initial_portfolio_value",
                    "daily_mtm",
                    "end_portfolio_value",
                    # Position details
                    "position_qty",
                    "position_avg_cost",
                    # Account summary
                    "total_value",
                    "num_positions",
                ]
            )

            for snapshot in backtest_obj.portfolio_snapshots:
                writer.writerow(
                    [
                        snapshot["timestamp"],
                        snapshot.get("symbol", ""),
                        # Bar OHLC data
                        snapshot.get("open", 0.0),
                        snapshot.get("high", 0.0),
                        snapshot.get("low", 0.0),
                        snapshot.get("close", 0.0),
                        snapshot.get("volume", 0),
                        # Adjustment factors
                        snapshot.get("cumulative_price_factor", ""),
                        snapshot.get("cumulative_volume_factor", ""),
                        snapshot.get("adjustment_factor", ""),
                        snapshot.get("adjustment_reason", ""),
                        # Fill information
                        snapshot.get("signal", ""),
                        snapshot.get("order_id", ""),
                        snapshot.get("order_type", ""),
                        snapshot.get("fill_qty", 0),
                        snapshot.get("fill_price", 0.0),
                        snapshot.get("commission", 0.0),
                        # Cash flow tracking
                        snapshot.get("initial_cash", 0.0),
                        snapshot.get("cash_debits", 0.0),
                        snapshot.get("cash_credits", 0.0),
                        snapshot.get("end_cash", 0.0),
                        # Portfolio tracking
                        snapshot.get("initial_portfolio_value", 0.0),
                        snapshot.get("daily_mtm", 0.0),
                        snapshot.get("end_portfolio_value", 0.0),
                        # Position details
                        snapshot.get("position_qty", 0),
                        snapshot.get("position_avg_cost", 0.0),
                        # Account summary
                        snapshot.get("total_value", 0.0),
                        snapshot.get("num_positions", 0),
                    ]
                )

        logger.info("exported_portfolio", path=str(snapshots_file), count=len(backtest_obj.portfolio_snapshots))

    # TODO: Implement remaining file types based on generate_files config:
    # - trades.csv (generate["trades"])
    # - positions.csv (generate["positions"])
    # - equity_curve.csv (generate["equity_curve"])


def _export_debug_files(backtest_obj: Backtest, output_dir: Path):
    """
    Export debug files (fills, snapshots, etc.).

    Args:
        backtest_obj: Backtest instance with results
        output_dir: Directory to write files
    """
    import csv

    # Export fills
    if backtest_obj.all_fills:
        fills_file = output_dir / "fills.csv"
        with open(fills_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fill_id",
                    "order_id",
                    "timestamp",
                    "symbol",
                    "side",
                    "qty",
                    "price",
                    "fees",
                    "slippage_bps",
                ]
            )

            for fill in backtest_obj.all_fills:
                writer.writerow(
                    [
                        fill.fill_id,
                        fill.order_id,
                        fill.execution_ts,
                        fill.symbol,
                        fill.side.name,
                        fill.qty,
                        fill.price,
                        fill.fees,
                        fill.slippage_bps,
                    ]
                )

        logger.info("exported_fills", path=str(fills_file), count=len(backtest_obj.all_fills))

    # Export portfolio snapshots
    if backtest_obj.portfolio_snapshots:
        snapshots_file = output_dir / "portfolio_snapshots.csv"
        with open(snapshots_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "cash",
                    "equity",
                    "total_value",
                    "num_positions",
                ]
            )

            for snapshot in backtest_obj.portfolio_snapshots:
                writer.writerow(
                    [
                        snapshot["timestamp"],
                        snapshot["cash"],
                        snapshot["equity"],
                        snapshot["total_value"],
                        snapshot["num_positions"],
                    ]
                )

        logger.info("exported_snapshots", path=str(snapshots_file), count=len(backtest_obj.portfolio_snapshots))


if __name__ == "__main__":
    main()
