# Phase 9: ReportingService

## Overview

**Goal:** Create a reporting service that formats and displays analytics results through multiple output channels (console, JSON, CSV, plots).

**Duration:** 1-2 weeks **Complexity:** Low-Medium **Priority:** High - User-facing output layer

**Depends On:** Phase 8 (AnalyticsService) for metrics calculation

## Rationale: Why Separate from Analytics?

### Clear Separation of Concerns

- **AnalyticsService** (Phase 8): COMPUTE metrics (pure math)
- **ReportingService** (Phase 9): DISPLAY results (formatting, plots, files)

### Benefits

- **Testability**: Test formatting independently of calculations
- **Flexibility**: Swap output formats without touching analytics
- **Reusability**: Analytics can be used without reporting
- **Single Responsibility**: Reporting only handles presentation

### Current Gap

Currently in master:

- Backtest results printed directly to console
- CSV exports hard-coded in backtest engine
- No standardized metrics calculation
- No programmatic access to results
- Plotting is manual/external

## Target Architecture

### Service Interface

```python
# src/qtrader/services/reporting/interface.py

from abc import abstractmethod
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Protocol

from qtrader.models.portfolio import Portfolio


class BacktestResult:
    """
    Backtest result data class.

    Contains all information needed for reporting:
    - Portfolio snapshots over time
    - Trade history
    - Performance metrics
    - Configuration used
    """

    def __init__(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_cash: Decimal,
        final_equity: Decimal,
        snapshots: list[dict],
        trades: list[dict],
        config: dict,
    ):
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.final_equity = final_equity
        self.snapshots = snapshots
        self.trades = trades
        self.config = config


class PerformanceMetrics:
    """Performance metrics calculated from backtest results."""

    def __init__(
        self,
        total_return: float,
        annualized_return: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        max_drawdown_duration: int,
        win_rate: float,
        profit_factor: float,
        num_trades: int,
        avg_win: float,
        avg_loss: float,
        largest_win: float,
        largest_loss: float,
    ):
        self.total_return = total_return
        self.annualized_return = annualized_return
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.max_drawdown = max_drawdown
        self.max_drawdown_duration = max_drawdown_duration
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.num_trades = num_trades
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.largest_win = largest_win
        self.largest_loss = largest_loss


class IReportingService(Protocol):
    """
    Reporting service interface.

    Responsibilities:
    - Calculate performance metrics
    - Format results for display
    - Export results to files (JSON, CSV)
    - Generate plots (equity curve, drawdown)
    - Provide programmatic access to results

    Does NOT:
    - Execute trades (that's ExecutionService)
    - Manage portfolio (that's PortfolioService)
    - Make decisions (that's Strategy)
    """

    @abstractmethod
    def calculate_metrics(self, result: BacktestResult) -> PerformanceMetrics:
        """
        Calculate performance metrics from backtest result.

        Args:
            result: Backtest result with snapshots and trades

        Returns:
            Performance metrics
        """
        ...

    @abstractmethod
    def print_summary(
        self, result: BacktestResult, metrics: PerformanceMetrics
    ) -> None:
        """
        Print human-readable summary to console.

        Args:
            result: Backtest result
            metrics: Performance metrics
        """
        ...

    @abstractmethod
    def export_json(self, result: BacktestResult, output_path: Path) -> None:
        """
        Export results to JSON file.

        Args:
            result: Backtest result
            output_path: Output file path
        """
        ...

    @abstractmethod
    def export_csv(
        self, result: BacktestResult, output_dir: Path
    ) -> dict[str, Path]:
        """
        Export results to CSV files.

        Creates multiple CSV files:
        - snapshots.csv: Portfolio snapshots over time
        - trades.csv: Trade history
        - metrics.csv: Performance metrics

        Args:
            result: Backtest result
            output_dir: Output directory

        Returns:
            Dict mapping file type to path
        """
        ...

    @abstractmethod
    def plot_equity_curve(
        self, result: BacktestResult, output_path: Optional[Path] = None
    ) -> None:
        """
        Plot equity curve over time.

        Args:
            result: Backtest result
            output_path: Output file path (displays if None)
        """
        ...

    @abstractmethod
    def plot_drawdown(
        self, result: BacktestResult, output_path: Optional[Path] = None
    ) -> None:
        """
        Plot drawdown over time.

        Args:
            result: Backtest result
            output_path: Output file path (displays if None)
        """
        ...

    @abstractmethod
    def compare_strategies(
        self, results: list[BacktestResult], output_path: Optional[Path] = None
    ) -> None:
        """
        Compare multiple strategy results.

        Args:
            results: List of backtest results
            output_path: Output file path (displays if None)
        """
        ...
```

### Service Implementation

```python
# src/qtrader/services/reporting/service.py

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from qtrader.config.logging_config import LoggerFactory
from qtrader.services.reporting.interface import (
    BacktestResult,
    IReportingService,
    PerformanceMetrics,
)
from qtrader.services.reporting.metrics import MetricsCalculator
from qtrader.services.reporting.plots import PlotGenerator

logger = LoggerFactory.get_logger()


class ReportingService:
    """
    Concrete implementation of reporting service.

    Consumes backtest results and produces formatted output.
    """

    def __init__(self):
        """Initialize reporting service."""
        self.metrics_calculator = MetricsCalculator()
        self.plot_generator = PlotGenerator()

        logger.info("reporting_service.initialized")

    def calculate_metrics(self, result: BacktestResult) -> PerformanceMetrics:
        """Calculate performance metrics."""
        return self.metrics_calculator.calculate(result)

    def print_summary(
        self, result: BacktestResult, metrics: PerformanceMetrics
    ) -> None:
        """Print summary to console."""
        print("\n" + "=" * 60)
        print(f"Backtest Results: {result.strategy_name}")
        print("=" * 60)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Cash: ${result.initial_cash:,.2f}")
        print(f"Final Equity: ${result.final_equity:,.2f}")
        print(f"Total Return: {metrics.total_return:.2%}")
        print(f"Annualized Return: {metrics.annualized_return:.2%}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Max DD Duration: {metrics.max_drawdown_duration} days")
        print(f"\nTrades: {metrics.num_trades}")
        print(f"Win Rate: {metrics.win_rate:.2%}")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Avg Win: ${metrics.avg_win:,.2f}")
        print(f"Avg Loss: ${metrics.avg_loss:,.2f}")
        print(f"Largest Win: ${metrics.largest_win:,.2f}")
        print(f"Largest Loss: ${metrics.largest_loss:,.2f}")
        print("=" * 60 + "\n")

        logger.info(
            "reporting_service.summary_printed",
            strategy=result.strategy_name,
            total_return=metrics.total_return,
            sharpe=metrics.sharpe_ratio,
        )

    def export_json(self, result: BacktestResult, output_path: Path) -> None:
        """Export to JSON."""
        data = {
            "strategy_name": result.strategy_name,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_cash": float(result.initial_cash),
            "final_equity": float(result.final_equity),
            "snapshots": result.snapshots,
            "trades": result.trades,
            "config": result.config,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("reporting_service.json_exported", path=str(output_path))

    def export_csv(
        self, result: BacktestResult, output_dir: Path
    ) -> dict[str, Path]:
        """Export to CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Snapshots
        snapshots_df = pd.DataFrame(result.snapshots)
        snapshots_path = output_dir / "snapshots.csv"
        snapshots_df.to_csv(snapshots_path, index=False)
        paths["snapshots"] = snapshots_path

        # Trades
        trades_df = pd.DataFrame(result.trades)
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        paths["trades"] = trades_path

        # Metrics
        metrics = self.calculate_metrics(result)
        metrics_df = pd.DataFrame([vars(metrics)])
        metrics_path = output_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        paths["metrics"] = metrics_path

        logger.info(
            "reporting_service.csv_exported",
            output_dir=str(output_dir),
            files=list(paths.keys()),
        )

        return paths

    def plot_equity_curve(
        self, result: BacktestResult, output_path: Optional[Path] = None
    ) -> None:
        """Plot equity curve."""
        self.plot_generator.plot_equity_curve(result, output_path)

    def plot_drawdown(
        self, result: BacktestResult, output_path: Optional[Path] = None
    ) -> None:
        """Plot drawdown."""
        self.plot_generator.plot_drawdown(result, output_path)

    def compare_strategies(
        self, results: list[BacktestResult], output_path: Optional[Path] = None
    ) -> None:
        """Compare multiple strategies."""
        self.plot_generator.compare_strategies(results, output_path)
```

### Metrics Calculator

```python
# src/qtrader/services/reporting/metrics.py

import numpy as np
from datetime import datetime
from decimal import Decimal

from qtrader.services.reporting.interface import BacktestResult, PerformanceMetrics


class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    def calculate(self, result: BacktestResult) -> PerformanceMetrics:
        """Calculate all metrics."""
        returns = self._calculate_returns(result.snapshots)

        total_return = self._total_return(result)
        annualized_return = self._annualized_return(result, total_return)
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        max_dd, max_dd_duration = self._max_drawdown(result.snapshots)

        trade_metrics = self._trade_metrics(result.trades)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            **trade_metrics,
        )

    def _calculate_returns(self, snapshots: list[dict]) -> np.ndarray:
        """Calculate daily returns."""
        equity = np.array([s["equity"] for s in snapshots])
        returns = np.diff(equity) / equity[:-1]
        return returns

    def _total_return(self, result: BacktestResult) -> float:
        """Total return percentage."""
        return (
            float(result.final_equity) / float(result.initial_cash) - 1.0
        )

    def _annualized_return(self, result: BacktestResult, total_return: float) -> float:
        """Annualized return."""
        days = (result.end_date - result.start_date).days
        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def _sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Sharpe ratio (annualized)."""
        excess_returns = returns - (risk_free_rate / 252)
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Sortino ratio (annualized)."""
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def _max_drawdown(self, snapshots: list[dict]) -> tuple[float, int]:
        """Maximum drawdown and duration."""
        equity = np.array([s["equity"] for s in snapshots])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        max_dd = float(np.min(drawdown))

        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        dd_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, dd_duration

    def _trade_metrics(self, trades: list[dict]) -> dict:
        """Calculate trade-based metrics."""
        if not trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(trades) if trades else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses)) if losses else float("inf")
        )

        return {
            "num_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "largest_win": max(wins) if wins else 0.0,
            "largest_loss": min(losses) if losses else 0.0,
        }
```

## Implementation Tasks

### Week 1: Core Reporting

- [ ] Create service structure
- [ ] Define `IReportingService` protocol
- [ ] Implement `ReportingService` class
- [ ] Implement `MetricsCalculator`
- [ ] Console output formatting
- [ ] JSON export

### Week 2: Advanced Reporting

- [ ] CSV export (snapshots, trades, metrics)
- [ ] Implement `PlotGenerator`
- [ ] Equity curve plot
- [ ] Drawdown plot
- [ ] Strategy comparison plots

### Week 3: Integration & Testing

- [ ] Integrate with BacktestEngine
- [ ] Unit tests for metrics calculations
- [ ] Integration tests with real backtest data
- [ ] Documentation
- [ ] Examples

## Testing Strategy

```python
def test_metrics_calculation():
    """Test metrics calculator."""
    result = create_test_result(
        initial_cash=100000,
        final_equity=150000,
        snapshots=[...],
        trades=[...],
    )

    service = ReportingService()
    metrics = service.calculate_metrics(result)

    assert metrics.total_return == 0.5  # 50% return
    assert metrics.sharpe_ratio > 0
    assert metrics.max_drawdown < 0
```

## Validation Criteria

- [ ] ✅ Calculate all standard metrics
- [ ] ✅ Console output formatted
- [ ] ✅ JSON export works
- [ ] ✅ CSV export works (3 files)
- [ ] ✅ Plots generated correctly
- [ ] ✅ Strategy comparison works
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ Documentation complete

## Dependencies

### Depends On

- Phase 2: PortfolioService (snapshots)
- Phase 5: BacktestEngine (orchestrates, produces results)

### Consumed By

- Users (console output, files)
- Future: OptimizationService (programmatic access)

## Success Metrics

- [ ] ✅ Can generate reports without manual code
- [ ] ✅ Results serializable to JSON
- [ ] ✅ Plots publication-ready
- [ ] ✅ Metrics calculation validated against known formulas

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phase 2 (Portfolio), Phase 5 (Backtest) **Estimated Duration:** 2-3 weeks **Last Updated:** October 15, 2025
