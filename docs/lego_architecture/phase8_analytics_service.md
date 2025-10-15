# Phase 8: AnalyticsService

## Overview

**Goal:** Create an analytics service that computes performance metrics and portfolio statistics from backtest results, independent of presentation layer.

**Duration:** 2 weeks **Complexity:** Medium **Priority:** High - Essential for strategy evaluation

## Rationale: Why Separate from Reporting?

### Clear Separation of Concerns

- **AnalyticsService**: COMPUTE metrics (Sharpe, drawdown, etc.)
- **ReportingService** (Phase 9): DISPLAY/EXPORT results (console, JSON, plots)

### Reusability

Multiple services need analytics:

- **OptimizationService**: Compare parameter sets by Sharpe ratio
- **LiveTradingService**: Monitor real-time strategy performance
- **ReportingService**: Display metrics in reports
- **User Code**: Programmatic access to metrics

### Testability

Test metric calculations independently:

```python
def test_sharpe_ratio():
    analytics = AnalyticsService()
    sharpe = analytics.calculate_sharpe(returns)
    assert abs(sharpe - 1.5) < 0.01  # Verify math is correct
```

## Target Architecture

### Service Interface

```python
# src/qtrader/services/analytics/interface.py

from abc import abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Protocol

import numpy as np


class PerformanceMetrics:
    """
    Performance metrics data class.

    All metrics calculated from backtest results.
    """

    def __init__(
        self,
        # Returns
        total_return: float,
        annualized_return: float,
        cagr: float,
        # Risk-adjusted returns
        sharpe_ratio: float,
        sortino_ratio: float,
        calmar_ratio: float,
        # Risk metrics
        volatility: float,
        downside_deviation: float,
        max_drawdown: float,
        max_drawdown_duration: int,
        avg_drawdown: float,
        # Trade statistics
        num_trades: int,
        win_rate: float,
        loss_rate: float,
        avg_win: float,
        avg_loss: float,
        largest_win: float,
        largest_loss: float,
        avg_win_loss_ratio: float,
        profit_factor: float,
        # Time in market
        avg_trade_duration: float,
        max_trade_duration: int,
        # Additional
        expectancy: float,
        kelly_criterion: float,
    ):
        self.total_return = total_return
        self.annualized_return = annualized_return
        self.cagr = cagr
        self.sharpe_ratio = sharpe_ratio
        self.sortino_ratio = sortino_ratio
        self.calmar_ratio = calmar_ratio
        self.volatility = volatility
        self.downside_deviation = downside_deviation
        self.max_drawdown = max_drawdown
        self.max_drawdown_duration = max_drawdown_duration
        self.avg_drawdown = avg_drawdown
        self.num_trades = num_trades
        self.win_rate = win_rate
        self.loss_rate = loss_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.largest_win = largest_win
        self.largest_loss = largest_loss
        self.avg_win_loss_ratio = avg_win_loss_ratio
        self.profit_factor = profit_factor
        self.avg_trade_duration = avg_trade_duration
        self.max_trade_duration = max_trade_duration
        self.expectancy = expectancy
        self.kelly_criterion = kelly_criterion


class BacktestResult:
    """
    Backtest result data class.

    Raw data from backtest execution.
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


class IAnalyticsService(Protocol):
    """
    Analytics service interface.

    Responsibilities:
    - Calculate performance metrics from backtest results
    - Compute risk-adjusted returns
    - Analyze trade statistics
    - Provide statistical analysis tools

    Does NOT:
    - Format output (that's ReportingService)
    - Make trading decisions (that's Strategy)
    - Execute trades (that's ExecutionService)
    """

    @abstractmethod
    def calculate_metrics(self, result: BacktestResult) -> PerformanceMetrics:
        """
        Calculate all performance metrics from backtest result.

        Args:
            result: Backtest result with snapshots and trades

        Returns:
            Performance metrics
        """
        ...

    @abstractmethod
    def calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio (annualized).

        Args:
            returns: Array of daily returns
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        ...

    @abstractmethod
    def calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino ratio (annualized).

        Args:
            returns: Array of daily returns
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sortino ratio
        """
        ...

    @abstractmethod
    def calculate_max_drawdown(
        self, equity_curve: np.ndarray
    ) -> tuple[float, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            equity_curve: Array of equity values over time

        Returns:
            Tuple of (max_drawdown, duration_in_days)
        """
        ...

    @abstractmethod
    def calculate_calmar_ratio(
        self, annualized_return: float, max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            annualized_return: Annualized return
            max_drawdown: Maximum drawdown (absolute value)

        Returns:
            Calmar ratio
        """
        ...

    @abstractmethod
    def calculate_win_rate(self, trades: list[dict]) -> float:
        """
        Calculate win rate from trades.

        Args:
            trades: List of trade dicts with 'pnl' field

        Returns:
            Win rate (0.0 to 1.0)
        """
        ...

    @abstractmethod
    def calculate_profit_factor(self, trades: list[dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade dicts with 'pnl' field

        Returns:
            Profit factor
        """
        ...

    @abstractmethod
    def calculate_kelly_criterion(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion for position sizing.

        Args:
            win_rate: Win rate (0.0 to 1.0)
            avg_win_loss_ratio: Average win / average loss

        Returns:
            Kelly percentage (0.0 to 1.0)
        """
        ...

    @abstractmethod
    def calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate daily returns from equity curve.

        Args:
            equity_curve: Array of equity values

        Returns:
            Array of daily returns
        """
        ...
```

### Service Implementation

```python
# src/qtrader/services/analytics/service.py

import numpy as np
from datetime import datetime
from decimal import Decimal

from qtrader.config.logging_config import LoggerFactory
from qtrader.services.analytics.interface import (
    BacktestResult,
    IAnalyticsService,
    PerformanceMetrics,
)

logger = LoggerFactory.get_logger()


class AnalyticsService:
    """
    Concrete implementation of analytics service.

    Computes all performance metrics from backtest results.
    Pure calculation logic, no formatting or display.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize analytics service.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info("analytics_service.initialized", risk_free_rate=risk_free_rate)

    def calculate_metrics(self, result: BacktestResult) -> PerformanceMetrics:
        """Calculate all metrics from backtest result."""
        logger.debug(
            "analytics_service.calculating_metrics",
            strategy=result.strategy_name,
            num_snapshots=len(result.snapshots),
            num_trades=len(result.trades),
        )

        # Extract data
        equity_curve = np.array([s["equity"] for s in result.snapshots])
        returns = self.calculate_returns(equity_curve)

        # Return metrics
        total_return = self._total_return(result)
        annualized_return = self._annualized_return(result, total_return)
        cagr = annualized_return  # Same as annualized return

        # Risk-adjusted returns
        sharpe = self.calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = self.calculate_sortino_ratio(returns, self.risk_free_rate)
        max_dd, max_dd_duration = self.calculate_max_drawdown(equity_curve)
        calmar = self.calculate_calmar_ratio(annualized_return, abs(max_dd))

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
        downside_returns = returns[returns < 0]
        downside_deviation = (
            np.std(downside_returns) * np.sqrt(252)
            if len(downside_returns) > 0
            else 0.0
        )
        avg_drawdown = self._average_drawdown(equity_curve)

        # Trade statistics
        trade_stats = self._calculate_trade_stats(result.trades)

        # Kelly criterion
        kelly = self.calculate_kelly_criterion(
            trade_stats["win_rate"],
            trade_stats["avg_win_loss_ratio"],
        )

        logger.info(
            "analytics_service.metrics_calculated",
            strategy=result.strategy_name,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=trade_stats["win_rate"],
        )

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_drawdown,
            kelly_criterion=kelly,
            **trade_stats,
        )

    def calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        if np.std(excess_returns) == 0:
            return 0.0

        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    def calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (annualized)."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        return float(
            np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        )

    def calculate_max_drawdown(
        self, equity_curve: np.ndarray
    ) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(equity_curve) == 0:
            return 0.0, 0

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max

        max_dd = float(np.min(drawdown))

        # Calculate duration
        in_drawdown = drawdown < -0.001  # Consider drawdowns > 0.1%
        dd_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                dd_duration = max(dd_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, dd_duration

    def calculate_calmar_ratio(
        self, annualized_return: float, max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio."""
        if abs(max_drawdown) < 0.0001:  # Avoid division by zero
            return 0.0
        return annualized_return / abs(max_drawdown)

    def calculate_win_rate(self, trades: list[dict]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        return wins / len(trades)

    def calculate_profit_factor(self, trades: list[dict]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0

        gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(
            sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0)
        )

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_kelly_criterion(
        self, win_rate: float, avg_win_loss_ratio: float
    ) -> float:
        """Calculate Kelly criterion."""
        if avg_win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        return max(0.0, min(kelly, 1.0))  # Clamp between 0 and 1

    def calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate daily returns."""
        if len(equity_curve) < 2:
            return np.array([])

        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    def _total_return(self, result: BacktestResult) -> float:
        """Calculate total return percentage."""
        return float(result.final_equity) / float(result.initial_cash) - 1.0

    def _annualized_return(self, result: BacktestResult, total_return: float) -> float:
        """Calculate annualized return."""
        days = (result.end_date - result.start_date).days
        years = days / 365.25
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    def _average_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate average drawdown."""
        if len(equity_curve) == 0:
            return 0.0

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        drawdowns = drawdown[drawdown < 0]

        return float(np.mean(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _calculate_trade_stats(self, trades: list[dict]) -> dict:
        """Calculate trade statistics."""
        if not trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_win_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_duration": 0.0,
                "max_trade_duration": 0,
                "expectancy": 0.0,
            }

        pnls = [t.get("pnl", 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        profit_factor = self.calculate_profit_factor(trades)

        # Trade duration (if available)
        durations = [
            t.get("duration", 0) for t in trades if "duration" in t
        ]
        avg_duration = np.mean(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0

        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        return {
            "num_trades": len(trades),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_duration,
            "max_trade_duration": max_duration,
            "expectancy": expectancy,
        }
```

## Implementation Tasks

### Week 1: Core Metrics

- [ ] Create service structure
  - `src/qtrader/services/analytics/__init__.py`
  - `src/qtrader/services/analytics/interface.py`
  - `src/qtrader/services/analytics/service.py`
- [ ] Define `IAnalyticsService` protocol
- [ ] Define `PerformanceMetrics` and `BacktestResult` data classes
- [ ] Implement core metrics:
  - [ ] Returns (total, annualized, CAGR)
  - [ ] Sharpe ratio
  - [ ] Sortino ratio
  - [ ] Maximum drawdown

### Week 2: Advanced Metrics & Testing

- [ ] Implement advanced metrics:
  - [ ] Calmar ratio
  - [ ] Volatility metrics
  - [ ] Trade statistics (win rate, profit factor)
  - [ ] Kelly criterion
- [ ] Write comprehensive unit tests
- [ ] Validate formulas against known benchmarks
- [ ] Performance optimization
- [ ] Documentation

## Testing Strategy

### Unit Tests (Pure Math)

```python
def test_sharpe_ratio_calculation():
    """Test Sharpe ratio with known data."""
    analytics = AnalyticsService(risk_free_rate=0.02)

    # Known returns with expected Sharpe
    returns = np.array([0.01, -0.005, 0.015, 0.008, -0.002])
    sharpe = analytics.calculate_sharpe_ratio(returns)

    # Verify against hand-calculated value
    assert abs(sharpe - 1.23) < 0.01


def test_max_drawdown():
    """Test maximum drawdown calculation."""
    analytics = AnalyticsService()

    equity_curve = np.array([100, 110, 105, 95, 100, 115])
    max_dd, duration = analytics.calculate_max_drawdown(equity_curve)

    assert abs(max_dd - (-0.136)) < 0.001  # 13.6% drawdown from 110 to 95
    assert duration == 3  # 3 bars in drawdown
```

### Integration Tests

```python
def test_calculate_all_metrics():
    """Test full metrics calculation from backtest result."""
    analytics = AnalyticsService()

    result = create_test_backtest_result(
        initial_cash=100000,
        final_equity=150000,
        snapshots=[...],
        trades=[...],
    )

    metrics = analytics.calculate_metrics(result)

    assert metrics.total_return == 0.5
    assert metrics.sharpe_ratio > 0
    assert metrics.max_drawdown < 0
    assert 0 <= metrics.win_rate <= 1
```

## Validation Criteria

- [ ] ✅ All metrics match industry standard formulas
- [ ] ✅ Sharpe/Sortino validated against finance libraries
- [ ] ✅ Drawdown calculation matches reference implementations
- [ ] ✅ Test coverage ≥ 95% (pure math should be testable)
- [ ] ✅ All edge cases handled (empty data, zero division)
- [ ] ✅ Performance: < 100ms for 1 year of daily data
- [ ] ✅ Documentation with formula references

## Dependencies

### Depends On

- Phase 2: PortfolioService (provides snapshots)
- Phase 5: BacktestEngine (produces BacktestResult)
- NumPy (for calculations)

### Consumed By

- Phase 9: ReportingService (displays metrics)
- Future: OptimizationService (compares strategies)
- Future: LiveTradingService (real-time monitoring)

## Success Metrics

- [ ] ✅ Can calculate metrics independently of reporting
- [ ] ✅ Results match validated finance formulas
- [ ] ✅ Performance acceptable for large backtests
- [ ] ✅ Easy to add custom metrics
- [ ] ✅ Reusable by multiple consumers

## Next Phase

👉 **[Phase 9: ReportingService](phase9_reporting_service.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phase 2 (Portfolio), Phase 5 (Backtest) **Estimated Duration:** 2 weeks **Last Updated:** October 15, 2025
