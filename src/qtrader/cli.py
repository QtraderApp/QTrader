"""Command-line interface."""

import click


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
    "--data",
    type=click.Path(exists=True),
    required=False,
    help="Path to data configuration YAML (optional, uses defaults if omitted)",
)
@click.option("--out", type=click.Path(), required=True, help="Output directory for results")
@click.option("--set", "overrides", multiple=True, help="Override strategy config: --set param=value")
@click.option(
    "--warmup/--no-warmup",
    default=False,
    help="Enable warmup phase to build indicator state before trading (default: disabled)",
)
@click.option(
    "--warmup-bars",
    type=int,
    default=None,
    help="Explicit warmup period in bars (default: auto-detect from indicators)",
)
def backtest(strategy, data, out, overrides, warmup, warmup_bars):
    """
    Run a backtest with a self-contained strategy file.

    Strategy file must contain a Strategy class and optionally a config.
    Data config YAML contains system settings (data source, adapter, validation).

    Warmup Phase:
    - When enabled, processes initial bars to build indicator state
    - Auto-detects required bars from registered indicators (SMA, RSI, MACD, etc.)
    - Can override with --warmup-bars for explicit control
    - Strategy on_bar() NOT called during warmup
    - Strategy on_start() called after warmup completes

    Examples:

        # Basic usage
        qtrader backtest --strategy my_strategy.py --out results/

        # With data config
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/

        # With parameter overrides
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/ \\
          --set fast_period=10 --set position_size=200

        # With warmup (auto-detect period)
        qtrader backtest --strategy my_strategy.py --out results/ --warmup

        # With explicit warmup period
        qtrader backtest --strategy my_strategy.py --out results/ --warmup --warmup-bars 50
    """
    click.echo(f"Warmup: {warmup}, Warmup Bars: {warmup_bars}")
    click.echo("Stage 1: CLI stub only")
    raise NotImplementedError("Full CLI implementation in later stages")


@main.command()
@click.option("--data", type=click.Path(exists=True), required=True, help="Path to data configuration YAML")
def validate_data(data):
    """
    Validate dataset without running backtest.

    Loads data according to config and validates:
    - All bars load successfully
    - OHLC relationships are valid
    - Frequency matches expected
    - No missing data gaps

    Example:

        qtrader validate-data --data algoseek.yaml
    """
    click.echo("Stage 1: CLI stub only")
    raise NotImplementedError("Full CLI implementation in later stages")


if __name__ == "__main__":
    main()
