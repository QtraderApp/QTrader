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
def backtest(strategy, data, out, overrides):
    """
    Run a backtest with a self-contained strategy file.

    Strategy file must contain a Strategy class and optionally a config.
    Data config YAML contains system settings (data source, adapter, validation).

    Examples:

        # Basic usage
        qtrader backtest --strategy my_strategy.py --out results/

        # With data config
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/

        # With parameter overrides
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/ \\
          --set fast_period=10 --set position_size=200
    """
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
