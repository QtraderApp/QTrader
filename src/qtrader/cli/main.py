"""QTrader CLI main entry point."""

import click

from qtrader.cli.commands import data_group


@click.group()
@click.version_option(version="0.1.0")
def main():
    """QTrader - Quantitative Trading Backtest System"""
    pass


# Register command groups
main.add_command(data_group)


if __name__ == "__main__":
    main()
