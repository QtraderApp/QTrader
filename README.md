# QTrader

**A modular, protocol-driven quantitative trading backtesting framework for Python.**

QTrader is designed for systematic traders, quantitative researchers, and algorithmic trading enthusiasts who need a robust, testable, and extensible platform for developing and backtesting trading strategies with real market data.

## 🎯 Project Overview

QTrader is a Python-based backtesting framework that enables you to:

- **Load multi-vendor market data** from sources like Algoseek, Schwab, and custom CSV files
- **Backtest trading strategies** with realistic execution simulation including slippage and commissions
- **Manage positions and portfolio state** with full accounting for cash flows and P&L
- **Handle corporate actions** (stock splits, dividends) with multiple adjustment modes
- **Stream large datasets efficiently** using iterator-based architecture
- **Develop independently testable components** using the "Lego Architecture" pattern

### Key Features

✅ **Multi-vendor data integration** - Pluggable adapter system for different data providers ✅ **Corporate actions handling** - Unadjusted, split-adjusted, and total return modes ✅ **Iterator-based streaming** - Memory-efficient for large datasets ✅ **Type-safe with Pydantic** - Runtime validation and IDE autocomplete ✅ **Comprehensive testing** - 577 tests with 90% coverage ✅ **CLI tools** - List datasets, update caches, browse raw data ✅ **Incremental cache updates** - Efficient data management ✅ **Protocol-driven design** - Easy to mock and test independently

## 🏗️ Design Philosophy

QTrader follows a **"Lego Architecture"** approach where each service is:

1. **Independent** - Services have single responsibilities and minimal dependencies
1. **Protocol-based** - Interfaces defined with Python Protocols for easy mocking
1. **Composable** - Services snap together like Lego blocks
1. **Testable** - Each service can be unit tested in isolation with mock dependencies
1. **Type-safe** - Full type hints for IDE support and runtime validation

### Core Principles

- **Separation of Concerns** - Data, portfolio, execution, and strategy logic are separate
- **Dependency Injection** - Services receive dependencies via constructors (protocol interfaces)
- **Configuration-Driven** - YAML-based configuration for data sources and system settings
- **Iterator-Based** - Streaming data processing to handle large datasets efficiently
- **Test-First Development** - High test coverage (90%+) for production reliability

## 🧱 Architecture Overview

QTrader is organized into distinct layers, each with specific responsibilities:

### 1. **Services Layer** (Independent Lego Blocks)

```
services/
├── data/          # DataService - Load, transform, stream market data
├── portfolio/     # PortfolioService - Track positions, cash, P&L (Phase 2)
├── execution/     # ExecutionService - Simulate order fills (Phase 3)
├── risk/          # RiskService - Validate orders, enforce limits (Phase 4)
├── backtest/      # BacktestEngine - Orchestrate backtests (Phase 5)
├── strategy/      # StrategyContext - User-facing strategy API (Phase 6)
├── indicators/    # IndicatorService - Technical indicators (Phase 7)
├── analytics/     # AnalyticsService - Performance metrics (Phase 8)
└── reporting/     # ReportingService - Generate reports (Phase 9)
```

**Current Status:** Phase 1 (DataService) ✅ Complete | Phase 2 (PortfolioService) 🚧 In Progress

### 2. **Models Layer** (Shared Data Contracts)

```
models/
├── bar.py           # Bar, PriceSeries (canonical OHLCV)
├── multi_bar.py     # MultiBar (3 adjustment modes)
├── instrument.py    # Instrument metadata
├── order.py         # Order, OrderType, OrderStatus
├── position.py      # Position (holdings)
├── trade.py         # Trade (executed orders)
├── portfolio.py     # Portfolio aggregate view
└── vendors/         # Vendor-specific models (Algoseek, Schwab)
```

### 3. **Adapters Layer** (Data Source Integration)

```
adapters/
├── algoseek.py      # Algoseek OHLC adapter
├── schwab.py        # Schwab API adapter (OAuth2, caching)
└── resolver.py      # Maps logical datasets → physical sources
```

### 4. **Data Layer** (Infrastructure)

```
data/
├── loader.py        # DataLoader (coordinates adapter + transform)
├── iterator.py      # PriceSeriesIterator (streaming)
├── bar_merger.py    # BarMerger (multi-symbol alignment)
└── dataset_updater.py  # Incremental cache updates
```

### 5. **Configuration Layer**

```
config/
├── data_config.py       # Data source configuration
├── system_config.py     # System-wide settings
├── portfolio_config.py  # Portfolio settings
└── ...                  # More configs per service
```

### 6. **CLI Layer** (Command-Line Tools)

```bash
qtrader data list              # List available datasets
qtrader data update            # Update cached data
qtrader data cache-info        # View cache status
qtrader data raw               # Browse raw data files
```

## 📊 Data Flow Example

```
1. User Strategy
   └─> StrategyContext (Phase 6)
       ├─> DataService (Phase 1) ──> Load AAPL, MSFT bars
       ├─> IndicatorService (Phase 7) ──> Calculate SMA(20)
       └─> Places order via context.buy("AAPL", 100)

2. BacktestEngine (Phase 5)
   ├─> Receives order from strategy
   ├─> RiskService (Phase 4) ──> Validates order (size, limits)
   ├─> ExecutionService (Phase 3) ──> Simulates fill (slippage)
   └─> PortfolioService (Phase 2) ──> Updates positions, cash

3. End of Backtest
   └─> AnalyticsService (Phase 8) ──> Calculate Sharpe, drawdown
       └─> ReportingService (Phase 9) ──> Generate HTML report
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/QtraderApp/QTrader.git
cd QTrader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from qtrader.services import DataService
from qtrader.config import DataConfig
from datetime import date

# Configure data service
config = DataConfig(
    dataset="schwab-us-equity-1d-adjusted",
    adjustment_mode="adjusted"
)

# Create service
data_service = DataService(config)

# Load data
iterator = data_service.load_symbol(
    "AAPL",
    start_date=date(2020, 1, 1),
    end_date=date(2020, 12, 31)
)

# Stream bars
for bar in iterator:
    print(f"{bar.timestamp}: ${bar.adjusted.close:.2f}")
```

### Using the CLI

```bash
# List available datasets
qtrader data list --verbose

# Update cached data for specific symbols
qtrader data update --dataset schwab-us-equity-1d-adjusted \
                    --symbols AAPL MSFT GOOGL

# View cache status
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
```

## 🧪 Testing

QTrader has comprehensive test coverage across all layers:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/qtrader --cov-report=html

# Run specific test suite
pytest tests/unit/services/data/

# Run integration tests
pytest tests/integration/
```

**Current Status:** 577 tests passing | 90% overall coverage

## 📚 Documentation

- **[Lego Architecture Overview](docs/lego_architecture/)** - Design philosophy and phase breakdown
- **[Phase 1: DataService](docs/lego_architecture/PHASE_1_COMPLETE.md)** - Complete implementation guide
- **[Data CLI User Guide](docs/DATA_CLI_USER_GUIDE.md)** - Command-line tools documentation
- **[API Design](docs/API_DESIGN.md)** - Service interfaces and contracts
- **[Data Update Guide](docs/DATA_UPDATE_GUIDE.md)** - Cache management and updates

## 🛣️ Roadmap

QTrader is being built in 10 phases following the Lego Architecture:

- ✅ **Phase 1** - DataService (Complete)
- 🚧 **Phase 2** - PortfolioService (In Progress)
- 📋 **Phase 3** - ExecutionService
- 📋 **Phase 4** - RiskService
- 📋 **Phase 5** - BacktestEngine
- 📋 **Phase 6** - StrategyContext
- 📋 **Phase 7** - IndicatorService
- 📋 **Phase 8** - AnalyticsService
- 📋 **Phase 9** - ReportingService
- 📋 **Phase 10** - ConfigurationService & Migration

See [TODO.md](docs/TODO.md) for detailed roadmap.

## 🤝 Contributing

Contributions are welcome! Please see our contribution guidelines (coming soon).

## 📝 License

[License TBD]

## 🙏 Acknowledgments

- Built with data from [Algoseek](https://www.algoseek.com/) and [Schwab](https://www.schwab.com/)
- Inspired by best practices from Zipline, Backtrader, and QuantConnect

______________________________________________________________________

**Status:** Active Development | **Branch:** `feature/lego-architecture` | **Last Updated:** October 2025
