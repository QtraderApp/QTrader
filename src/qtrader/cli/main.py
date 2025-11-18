"""QTrader CLI main entry point."""

import click

from qtrader import __version__
from qtrader.cli.commands import data_group, init_library_command, init_project_command
from qtrader.cli.commands.backtest import backtest_command


@click.group()
@click.version_option(version=__version__)
def main():
    """QTrader - Quantitative Trading Backtest System"""
    pass


# Register commands
main.add_command(data_group)
main.add_command(backtest_command)
main.add_command(init_library_command)
main.add_command(init_project_command)


if __name__ == "__main__":
    main()
