# Logging Configuration Guide

## Overview

QTrader uses a centralized logging system based on `structlog` for structured, production-ready logging. The `LoggerFactory` provides a consistent way to configure and obtain logger instances across the entire codebase.

## Quick Start

### Basic Usage

```python
from qtrader.config import LoggerFactory

# Get a logger (auto-configures with defaults on first use)
logger = LoggerFactory.get_logger()

# Log structured messages
logger.info("trading.order_placed", symbol="AAPL", quantity=100, price=150.25)
logger.warning("market.volatility_high", vix_level=28.5)
logger.error("execution.order_rejected", order_id=12345, reason="insufficient_funds")
```

### Explicit Configuration

```python
from pathlib import Path
from qtrader.config import LoggerFactory, LoggingConfig

# Configure logging at application startup
config = LoggingConfig(
    level="DEBUG",                    # Minimum log level
    format="console",                 # Human-readable output
    enable_file=True,                 # Write logs to file
    file_path=Path("logs/qtrader.log"),
    file_rotation=True,               # Rotate when file gets large
    max_file_size_mb=100,            # Max size before rotation
    backup_count=5                    # Keep 5 rotated files
)

LoggerFactory.configure(config)

# Now all loggers use this configuration
logger = LoggerFactory.get_logger()
```

## Configuration Options

### LoggingConfig Parameters

| Parameter          | Type   | Default     | Description                                                               |
| ------------------ | ------ | ----------- | ------------------------------------------------------------------------- |
| `level`            | `str`  | `"INFO"`    | Minimum log level: DEBUG, INFO, WARNING, ERROR, CRITICAL                  |
| `format`           | `str`  | `"console"` | Output format: "console" (colored, human-readable) or "json" (structured) |
| `enable_file`      | `bool` | `False`     | Enable logging to file                                                    |
| `file_path`        | `Path` | `None`      | Path to log file (required if enable_file=True)                           |
| `file_level`       | `str`  | `"DEBUG"`   | Minimum log level for file output                                         |
| `file_rotation`    | `bool` | `True`      | Enable log file rotation                                                  |
| `max_file_size_mb` | `int`  | `100`       | Maximum file size in MB before rotation                                   |
| `backup_count`     | `int`  | `5`         | Number of rotated files to keep                                           |

## Output Formats

### Console Format

Human-readable with color coding:

```
2025-10-03T15:30:45.123456Z [info    ] trading.order_placed       [qtrader.execution] symbol=AAPL quantity=100 price=150.25
2025-10-03T15:30:46.234567Z [warning ] market.volatility_high     [qtrader.monitor] vix_level=28.5
2025-10-03T15:30:47.345678Z [error   ] execution.order_rejected   [qtrader.execution] order_id=12345 reason=insufficient_funds
```

### JSON Format

Machine-parseable structured logs:

```json
{"event": "trading.order_placed", "timestamp": "2025-10-03T15:30:45.123456Z", "level": "info", "logger": "qtrader.execution", "symbol": "AAPL", "quantity": 100, "price": 150.25}
{"event": "market.volatility_high", "timestamp": "2025-10-03T15:30:46.234567Z", "level": "warning", "logger": "qtrader.monitor", "vix_level": 28.5}
{"event": "execution.order_rejected", "timestamp": "2025-10-03T15:30:47.345678Z", "level": "error", "logger": "qtrader.execution", "order_id": 12345, "reason": "insufficient_funds"}
```

## Usage Patterns

### In Application Code

```python
# In your module
from qtrader.config import LoggerFactory

logger = LoggerFactory.get_logger()  # Auto-detects module name

class Strategy:
    def on_bar(self, bar):
        logger.debug("strategy.bar_received",
                    symbol=bar.symbol,
                    close=float(bar.close))

        if self.should_buy(bar):
            logger.info("strategy.signal_generated",
                       signal="BUY",
                       symbol=bar.symbol,
                       price=float(bar.close))
```

### With Explicit Logger Name

```python
from qtrader.config import LoggerFactory

# Use a specific logger name
logger = LoggerFactory.get_logger("myapp.custom_module")

logger.info("custom.event", key="value")
```

### Different Configurations for Different Environments

```python
import os
from pathlib import Path
from qtrader.config import LoggerFactory, LoggingConfig

# Development: verbose console logging
if os.getenv("ENV") == "development":
    config = LoggingConfig(
        level="DEBUG",
        format="console"
    )

# Production: JSON logging to file
elif os.getenv("ENV") == "production":
    config = LoggingConfig(
        level="INFO",
        format="json",
        enable_file=True,
        file_path=Path("/var/log/qtrader/app.log"),
        file_rotation=True,
        max_file_size_mb=500,
        backup_count=10
    )

# Testing: minimal logging
else:
    config = LoggingConfig(
        level="WARNING",
        format="console"
    )

LoggerFactory.configure(config)
```

## Best Practices

### 1. Use Structured Logging

Always use key-value pairs for context:

```python
# Good: Structured logging
logger.info("order.filled",
           order_id=12345,
           symbol="AAPL",
           quantity=100,
           price=150.25,
           timestamp=datetime.now().isoformat())

# Bad: Unstructured string formatting
logger.info(f"Order {order_id} filled: {quantity} shares of {symbol} @ ${price}")
```

### 2. Use Semantic Event Names

Use dot-notation for hierarchical event names:

```python
logger.info("trading.order.placed", ...)
logger.info("trading.order.filled", ...)
logger.info("trading.order.rejected", ...)
logger.info("market.data.received", ...)
logger.info("strategy.signal.generated", ...)
```

### 3. Choose Appropriate Log Levels

- **DEBUG**: Detailed diagnostic information (disable in production)
- **INFO**: Interesting events (order placed, signal generated)
- **WARNING**: Unexpected situations that don't break functionality
- **ERROR**: Errors that break functionality
- **CRITICAL**: System failures requiring immediate attention

```python
logger.debug("cache.lookup", key=key, found=True)
logger.info("trading.order_placed", order_id=12345)
logger.warning("api.rate_limit_approaching", remaining=10)
logger.error("database.connection_failed", error=str(e))
logger.critical("system.out_of_memory", available_mb=50)
```

### 4. Include Context

Add relevant context to make logs actionable:

```python
logger.error("order.execution_failed",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            account_balance=account.balance,
            error_code=error.code,
            error_message=str(error))
```

### 5. Configure Once at Startup

Configure logging once when your application starts:

```python
# main.py or __init__.py
from qtrader.config import LoggerFactory, LoggingConfig

def setup_logging():
    config = LoggingConfig(
        level="INFO",
        enable_file=True,
        file_path=Path("logs/qtrader.log")
    )
    LoggerFactory.configure(config)

if __name__ == "__main__":
    setup_logging()
    # Rest of application...
```

## Log File Management

### File Rotation

When `file_rotation=True`, log files automatically rotate when they reach `max_file_size_mb`:

```
logs/
  qtrader.log         # Current log file
  qtrader.log.1       # Previous rotation
  qtrader.log.2
  qtrader.log.3
  qtrader.log.4
  qtrader.log.5       # Oldest (then deleted on next rotation)
```

### Parsing JSON Logs

JSON format logs are easy to parse and analyze:

```python
import json

# Read and analyze JSON logs
with open("logs/qtrader.log") as f:
    for line in f:
        log = json.loads(line)
        if log["event"].startswith("trading.order"):
            print(f"{log['timestamp']}: {log['event']} - {log.get('symbol')}")
```

### Log Aggregation

Use tools like `jq` to query JSON logs:

```bash
# Count events by type
cat logs/qtrader.log | jq -r '.event' | sort | uniq -c

# Find all errors
cat logs/qtrader.log | jq 'select(.level == "error")'

# Get all orders for AAPL
cat logs/qtrader.log | jq 'select(.symbol == "AAPL")'
```

## Testing with Logging

### Reset Configuration in Tests

```python
import pytest
from qtrader.config import LoggerFactory

@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging between tests."""
    LoggerFactory.reset()
    yield
    LoggerFactory.reset()
```

### Capture Log Output

```python
import logging
from pathlib import Path
from qtrader.config import LoggerFactory, LoggingConfig

def test_my_feature(tmp_path):
    # Configure logging to temp file
    log_file = tmp_path / "test.log"
    config = LoggingConfig(
        level="DEBUG",
        enable_file=True,
        file_path=log_file
    )
    LoggerFactory.configure(config)

    # Run your test
    logger = LoggerFactory.get_logger()
    logger.info("test.event", key="value")

    # Verify logs
    content = log_file.read_text()
    assert "test.event" in content
```

## Migration from Direct structlog Usage

If you have existing code using `structlog.get_logger()` directly:

```python
# Old code
import structlog
logger = structlog.get_logger()

# New code
from qtrader.config import LoggerFactory
logger = LoggerFactory.get_logger()
```

The `LoggerFactory` provides the same interface but with centralized configuration.

## Troubleshooting

### Logs Not Appearing

- Check that `level` is set appropriately (DEBUG captures everything)
- Verify `file_path` directory exists and is writable
- Ensure `LoggerFactory.configure()` is called before any logging

### Colors Not Showing

- Use `format="console"` for colored output
- Colors may not work in some terminals (try `format="json"`)

### File Growing Too Large

- Enable `file_rotation=True`
- Reduce `max_file_size_mb`
- Reduce `backup_count` to keep fewer old files
- Use log level `INFO` or `WARNING` in production instead of `DEBUG`

## See Also

- [structlog documentation](https://www.structlog.org/)
- [Python logging documentation](https://docs.python.org/3/library/logging.html)
- [Best practices for application logging](https://www.structlog.org/en/stable/why.html)
