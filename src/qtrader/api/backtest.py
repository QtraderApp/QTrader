"""Backtest runner and config loader."""

from pathlib import Path


def load_config(path: Path):
    """Load configuration from YAML file."""
    raise NotImplementedError("Stage 1: Stub only")


class Backtest:
    """
    Backtest runner.

    Stub for Stage 1. Full implementation in later stages.
    """

    def __init__(self, config, strategy):
        self.config = config
        self.strategy = strategy

    def run(self, out_dir: Path):
        """Run backtest and write outputs."""
        raise NotImplementedError("Stage 1: Stub only")


def run_backtest(config_path: Path, strategy_class, out_dir: Path):
    """Convenience function to run backtest."""
    raise NotImplementedError("Stage 1: Stub only")
