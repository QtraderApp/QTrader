# QTrader Phase 1 — Implementation Plan

**Version:** 1.0\
**Date:** October 3, 2025\
**Status:** Ready for Implementation\
**Reference:** `docs/specs/phase01.md`

______________________________________________________________________

## 📊 Data Schema Analysis

### Sample Dataset Characteristics

**Location:** `data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/`

**Format:** Parquet partitioned by `SecId` (Hive-style: `SecId=33127/data_0.parquet`)

**Universe:**

- AAPL (SecId=33449)
- AMZN (SecId=33127)
- MSFT (SecId=39827)

**Date Range:** 2019-01-02 to 2023-12-29 (1,258 trading days)

**Schema:**

```python
TradeDate                 datetime64[ns]  # Trading date (no timezone in source)
Ticker                    str             # Symbol (e.g., "AAPL")
Open                      float64         # Opening price
High                      float64         # High price
Low                       float64         # Low price
Close                     float64         # Closing price
MarketHoursVolume         int64           # Volume during market hours
CumulativePriceFactor     float64         # Cumulative adjustment factor for prices
CumulativeVolumeFactor    float64         # Cumulative adjustment factor for volume
AdjustmentFactor          float64         # Period adjustment (usually NaN)
AdjustmentReason          str|None        # e.g., "CashDiv", None for no adjustment
SecId                     int64           # Algoseek security identifier
```

**Adjustment Events:**

- AAPL: 20 dividend events (CashDiv)
- MSFT: 20 dividend events (CashDiv)
- AMZN: 0 dividend events
- No splits in sample period

**Column Mapping (Algoseek → Bar):**

```yaml
ts: TradeDate
symbol: Ticker
open: Open
high: High
low: Low
close: Close
volume: MarketHoursVolume
adj_reason: AdjustmentReason
px_factor: CumulativePriceFactor
vol_factor: CumulativeVolumeFactor
```

______________________________________________________________________

## 🎯 Implementation Strategy

### Core Principles

1. **Decimal Precision:**

   - Bar prices (open/high/low/close): `Decimal` from adapter onward
   - Ledger (cash, PnL, costs): `Decimal`
   - Strategy indicators: `float64` for performance
   - Convert at adapter boundary and before indicators

1. **Data Adapters:**

   - **Primary:** Parquet adapter using DuckDB (matches fixture format)
   - **Secondary:** CSV adapter for security master linkage
   - Both adapters emit canonical `Bar` objects

1. **Testing Approach:**

   - TDD: Write tests first for each component
   - Focus on functional paths (not line coverage)
   - Use fixture data for all tests
   - Generate golden baselines in final stage

1. **Golden Baseline Generation:**

   - Create standalone scripts in `scripts/goldens/`
   - Run Buy-and-Hold and SMA Cross on fixture data
   - Manually validate results together
   - Commit golden files to `tests/goldens/fixtures/`
   - Automate validation in CI

______________________________________________________________________

## 🚀 Implementation Stages

### **Stage 1: Core Data Models & Parquet Adapter**

**Timeline:** Days 1-2\
**Branch:** `stage-1-data-foundation`

#### Deliverables

##### 1.1 Core Types (`src/models/bar.py`)

```python
from decimal import Decimal
from datetime import datetime
from typing import NamedTuple, Optional
from enum import Enum

class Bar(NamedTuple):
    """Canonical bar representing OHLCV data with adjustments."""
    ts: datetime                      # Timezone-aware timestamp
    symbol: str                       # Ticker symbol
    open: Decimal                     # Opening price
    high: Decimal                     # High price
    low: Decimal                      # Low price
    close: Decimal                    # Closing price
    volume: int                       # Volume (shares)
    adj_reason: Optional[str] = None  # Adjustment reason (e.g., "CashDiv")
    px_factor: Optional[Decimal] = None    # Cumulative price factor
    vol_factor: Optional[Decimal] = None   # Cumulative volume factor

class BarFrequency(Enum):
    """Supported bar frequencies."""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

class DataMode(Enum):
    """Data adjustment modes."""
    STANDARD_ADJUSTED = "standard_adjusted"  # Total-return adjusted

class OHLCPolicy(Enum):
    """Policies for handling malformed OHLC bars."""
    STRICT_RAISE = "strict_raise"          # Raise error on first violation
    WARN_SKIP_BAR = "warn_skip_bar"        # Warn and skip bar
    WARN_USE_CLOSE_ONLY = "warn_use_close_only"  # Warn and disable limit/stop
```

##### 1.2 Configuration Schema (`src/config/data_config.py`)

```python
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, Optional

class ValidationConfig(BaseModel):
    """OHLC validation configuration."""
    epsilon: float = Field(default=0.0, description="Tolerance for OHLC checks")
    ohlc_policy: str = Field(default="strict_raise")
    close_only_fields: list[str] = Field(default=["close"])

class DataConfig(BaseModel):
    """Data loading and processing configuration."""
    mode: str = Field(default="standard_adjusted")
    frequency: str = Field(default="1d")
    timezone: str = Field(default="America/New_York")
    strict_frequency: bool = Field(default=True)
    decimals: Dict[str, int] = Field(default={"price": 4, "cash": 4})
    source_tag: str = Field(default="algoseek-standard-adjusted")
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    column_map: Dict[str, str] = Field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("data", {}))
```

##### 1.3 Data Adapter Protocol (`src/adapters/base.py`)

```python
from typing import Protocol, Iterator, Any
from pathlib import Path
from src.models.bar import Bar
from src.config.data_config import DataConfig

class DataAdapter(Protocol):
    """Protocol for data adapters that normalize vendor data to canonical Bar."""
    
    def can_read(self, source: Path) -> bool:
        """Check if this adapter can read from the given source."""
        ...
    
    def schema_version(self) -> str:
        """Return the adapter schema version for reproducibility."""
        ...
    
    def read_iter(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """
        Read and normalize data from source.
        
        Yields canonical Bar objects with:
        - Decimal prices (quantized to config.decimals.price)
        - Timezone-aware timestamps
        - Validated OHLC relationships
        """
        ...
```

##### 1.4 Algoseek Parquet Adapter (`src/adapters/algoseek_parquet.py`)

```python
import duckdb
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from typing import Iterator
import pytz

from src.models.bar import Bar
from src.config.data_config import DataConfig
from src.adapters.base import DataAdapter

class AlgoseekParquetAdapter:
    """Adapter for Algoseek parquet data with Hive partitioning."""
    
    SCHEMA_VERSION = "algoseek-parquet-v1.0"
    
    # Default column mapping
    DEFAULT_COLUMN_MAP = {
        "ts": "TradeDate",
        "symbol": "Ticker",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "MarketHoursVolume",
        "adj_reason": "AdjustmentReason",
        "px_factor": "CumulativePriceFactor",
        "vol_factor": "CumulativeVolumeFactor",
    }
    
    def can_read(self, source: Path) -> bool:
        """Check if source contains parquet files."""
        if not source.exists():
            return False
        # Check for .parquet files or Hive-style partitions
        return any(source.rglob("*.parquet"))
    
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION
    
    def read_iter(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """
        Read parquet files and yield canonical Bar objects.
        
        Steps:
        1. Connect to DuckDB in-memory
        2. Read parquet with hive_partitioning=true
        3. Apply column mapping
        4. Convert types (float → Decimal, timestamp → tz-aware)
        5. Yield Bar objects in timestamp order
        """
        # Merge default and user column maps
        column_map = {**self.DEFAULT_COLUMN_MAP, **config.column_map}
        
        # Build parquet glob pattern
        parquet_pattern = str(source / "**" / "*.parquet")
        
        # Configure timezone
        tz = pytz.timezone(config.timezone)
        
        # Decimal quantization context
        price_decimals = config.decimals.get("price", 4)
        quantizer = Decimal(10) ** -price_decimals
        
        # Connect to DuckDB and read
        con = duckdb.connect(":memory:")
        try:
            # Build SELECT with column mapping
            select_cols = [
                f"{column_map['ts']} as ts",
                f"{column_map['symbol']} as symbol",
                f"{column_map['open']} as open",
                f"{column_map['high']} as high",
                f"{column_map['low']} as low",
                f"{column_map['close']} as close",
                f"{column_map['volume']} as volume",
                f"{column_map['adj_reason']} as adj_reason",
                f"{column_map['px_factor']} as px_factor",
                f"{column_map['vol_factor']} as vol_factor",
            ]
            
            query = f"""
            SELECT {', '.join(select_cols)}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            ORDER BY symbol, ts
            """
            
            result = con.execute(query)
            
            # Yield bars one at a time
            for row in result.fetchall():
                ts_naive, symbol, open_f, high_f, low_f, close_f, volume, adj_reason, px_f, vol_f = row
                
                # Localize timestamp
                ts = tz.localize(ts_naive)
                
                # Convert prices to Decimal
                open_d = Decimal(str(open_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                high_d = Decimal(str(high_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                low_d = Decimal(str(low_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                close_d = Decimal(str(close_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                
                # Convert factors to Decimal (or None)
                px_factor = Decimal(str(px_f)).quantize(Decimal("0.0000001")) if px_f is not None else None
                vol_factor = Decimal(str(vol_f)).quantize(Decimal("0.0000001")) if vol_f is not None else None
                
                yield Bar(
                    ts=ts,
                    symbol=symbol,
                    open=open_d,
                    high=high_d,
                    low=low_d,
                    close=close_d,
                    volume=int(volume),
                    adj_reason=adj_reason if adj_reason else None,
                    px_factor=px_factor,
                    vol_factor=vol_factor,
                )
        finally:
            con.close()
```

##### 1.5 CSV Adapter (`src/adapters/csv_adapter.py`)

```python
import pandas as pd
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterator
import pytz

from src.models.bar import Bar
from src.config.data_config import DataConfig

class CSVAdapter:
    """Adapter for CSV files (e.g., security master or exported CSVs)."""
    
    SCHEMA_VERSION = "csv-v1.0"
    
    def can_read(self, source: Path) -> bool:
        """Check if source is a CSV file or directory with CSVs."""
        if source.is_file() and source.suffix == ".csv":
            return True
        if source.is_dir():
            return any(source.glob("*.csv"))
        return False
    
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION
    
    def read_iter(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """Read CSV files and yield Bar objects."""
        # Column mapping (same default as Parquet adapter)
        column_map = config.column_map or {
            "ts": "TradeDate",
            "symbol": "Ticker",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "MarketHoursVolume",
        }
        
        tz = pytz.timezone(config.timezone)
        price_decimals = config.decimals.get("price", 4)
        quantizer = Decimal(10) ** -price_decimals
        
        # Determine CSV files to read
        csv_files = []
        if source.is_file():
            csv_files = [source]
        else:
            csv_files = sorted(source.glob("*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=[column_map["ts"]])
            
            for _, row in df.iterrows():
                ts_naive = pd.Timestamp(row[column_map["ts"]])
                ts = tz.localize(ts_naive.to_pydatetime())
                
                open_d = Decimal(str(row[column_map["open"]])).quantize(quantizer, ROUND_HALF_UP)
                high_d = Decimal(str(row[column_map["high"]])).quantize(quantizer, ROUND_HALF_UP)
                low_d = Decimal(str(row[column_map["low"]])).quantize(quantizer, ROUND_HALF_UP)
                close_d = Decimal(str(row[column_map["close"]])).quantize(quantizer, ROUND_HALF_UP)
                
                yield Bar(
                    ts=ts,
                    symbol=row[column_map["symbol"]],
                    open=open_d,
                    high=high_d,
                    low=low_d,
                    close=close_d,
                    volume=int(row[column_map["volume"]]),
                    adj_reason=row.get(column_map.get("adj_reason")),
                    px_factor=None,
                    vol_factor=None,
                )
```

##### 1.6 Bar Validator (`src/validation/bar_validator.py`)

```python
import logging
from decimal import Decimal
from typing import Optional
from datetime import timedelta

from src.models.bar import Bar, OHLCPolicy
from src.config.data_config import DataConfig

logger = logging.getLogger(__name__)

class BarValidator:
    """Validates Bar integrity and applies OHLC policies."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.epsilon = Decimal(str(config.validation.epsilon))
        self.policy = OHLCPolicy(config.validation.ohlc_policy)
        self.malformed_count = 0
        self.skipped_count = 0
        self.close_only_count = 0
    
    def validate_ohlc(self, bar: Bar) -> tuple[bool, Optional[str]]:
        """
        Validate OHLC relationships.
        
        Returns:
            (is_valid, reason) where reason is None if valid
        """
        # Check high >= max(open, close)
        if bar.high < max(bar.open, bar.close) - self.epsilon:
            return False, f"high ({bar.high}) < max(open, close)"
        
        # Check low <= min(open, close)
        if bar.low > min(bar.open, bar.close) + self.epsilon:
            return False, f"low ({bar.low}) > min(open, close)"
        
        # Check low <= high
        if bar.low > bar.high + self.epsilon:
            return False, f"low ({bar.low}) > high ({bar.high})"
        
        # Check volume >= 0
        if bar.volume < 0:
            return False, f"volume ({bar.volume}) < 0"
        
        return True, None
    
    def process_bar(self, bar: Bar) -> tuple[Optional[Bar], bool]:
        """
        Process bar according to OHLC policy.
        
        Returns:
            (bar_or_none, is_close_only)
            - bar_or_none: None if skipped, otherwise the bar
            - is_close_only: True if bar should only use close (no limit/stop)
        """
        is_valid, reason = self.validate_ohlc(bar)
        
        if is_valid:
            return bar, False
        
        # Malformed bar - apply policy
        self.malformed_count += 1
        
        if self.policy == OHLCPolicy.STRICT_RAISE:
            raise ValueError(
                f"Malformed OHLC bar at {bar.ts} for {bar.symbol}: {reason}"
            )
        
        elif self.policy == OHLCPolicy.WARN_SKIP_BAR:
            logger.warning(
                f"Skipping malformed bar: {bar.symbol} @ {bar.ts}: {reason}"
            )
            self.skipped_count += 1
            return None, False
        
        elif self.policy == OHLCPolicy.WARN_USE_CLOSE_ONLY:
            logger.warning(
                f"Close-only bar (limit/stop disabled): {bar.symbol} @ {bar.ts}: {reason}"
            )
            self.close_only_count += 1
            return bar, True
        
        return bar, False
    
    def validate_frequency(self, bars: list[Bar]) -> bool:
        """
        Validate that bars match expected frequency.
        
        Returns True if valid, raises ValueError if strict_frequency=true and invalid.
        """
        if not self.config.strict_frequency or len(bars) < 3:
            return True
        
        # Calculate median delta
        deltas = []
        for i in range(1, len(bars)):
            if bars[i].symbol == bars[i-1].symbol:
                delta = bars[i].ts - bars[i-1].ts
                deltas.append(delta)
        
        if not deltas:
            return True
        
        deltas.sort()
        median_delta = deltas[len(deltas) // 2]
        
        # Expected delta for frequency
        freq = self.config.frequency
        expected_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }
        
        expected = expected_deltas.get(freq)
        if expected and abs((median_delta - expected).total_seconds()) > 60:
            msg = f"Frequency mismatch: expected {freq}, median delta {median_delta}"
            if self.config.strict_frequency:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        
        return True
    
    def get_stats(self) -> dict:
        """Return validation statistics."""
        return {
            "malformed_bars": self.malformed_count,
            "skipped": self.skipped_count,
            "close_only": self.close_only_count,
        }
```

#### Tests (`tests/stage1/`)

```python
# tests/stage1/test_bar_model.py
def test_bar_creation_with_decimal_prices():
    """Bar should store prices as Decimal."""
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150.25"),
        high=Decimal("151.50"),
        low=Decimal("149.75"),
        close=Decimal("151.00"),
        volume=1000000,
    )
    assert isinstance(bar.open, Decimal)
    assert bar.open == Decimal("150.25")

# tests/stage1/test_algoseek_adapter.py
def test_adapter_reads_fixture_parquet():
    """Adapter should load AAPL data from fixture."""
    config = DataConfig(timezone="America/New_York")
    adapter = AlgoseekParquetAdapter()
    source = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")
    
    bars = list(adapter.read_iter(source, config))
    
    # Should have 3 symbols × 1258 days = 3774 bars
    assert len(bars) == 3774
    
    # Check first AAPL bar
    aapl_bars = [b for b in bars if b.symbol == "AAPL"]
    assert len(aapl_bars) == 1258
    assert aapl_bars[0].ts.date() == date(2019, 1, 2)
    assert isinstance(aapl_bars[0].close, Decimal)

def test_adapter_finds_adjustment_events():
    """Adapter should preserve adjustment reasons."""
    config = DataConfig()
    adapter = AlgoseekParquetAdapter()
    source = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")
    
    bars = list(adapter.read_iter(source, config))
    adj_bars = [b for b in bars if b.adj_reason is not None]
    
    # AAPL + MSFT have ~40 dividend events in 2019-2023
    assert len(adj_bars) > 30
    assert all(b.adj_reason == "CashDiv" for b in adj_bars)

# tests/stage1/test_bar_validator.py
def test_validator_strict_raise_on_malformed():
    """Validator should raise on malformed bar with strict policy."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="strict_raise"))
    validator = BarValidator(config)
    
    bad_bar = Bar(
        ts=datetime.now(pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("99"),  # Invalid: high < open
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1000,
    )
    
    with pytest.raises(ValueError, match="Malformed OHLC"):
        validator.process_bar(bad_bar)

def test_validator_warn_skip_bar():
    """Validator should skip bar and return None with warn_skip policy."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="warn_skip_bar"))
    validator = BarValidator(config)
    
    bad_bar = Bar(
        ts=datetime.now(pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("99"),
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1000,
    )
    
    result, is_close_only = validator.process_bar(bad_bar)
    assert result is None
    assert validator.skipped_count == 1

def test_validator_warn_use_close_only():
    """Validator should allow bar but flag as close-only."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="warn_use_close_only"))
    validator = BarValidator(config)
    
    bad_bar = Bar(
        ts=datetime.now(pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("99"),
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1000,
    )
    
    result, is_close_only = validator.process_bar(bad_bar)
    assert result is not None
    assert is_close_only is True
    assert validator.close_only_count == 1
```

#### Acceptance Criteria

- ✅ Load 3,774 bars from parquet fixture (3 symbols × 1,258 days)
- ✅ All prices stored as Decimal with 4 decimal places
- ✅ Timestamps timezone-aware (America/New_York)
- ✅ Adjustment events preserved (40 CashDiv events)
- ✅ All OHLC validation policies work correctly
- ✅ All Stage 1 tests pass (`make test`)

______________________________________________________________________

### **Stage 2: Order Models & Ledger Foundation**

**Timeline:** Days 3-4\
**Branch:** `stage-2-orders-ledger`

#### Deliverables

##### 2.1 Order Models (`src/models/order.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    MOC = "MOC"  # Market-On-Close
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderState(Enum):
    SUBMITTED = "SUBMITTED"
    TRIGGERED = "TRIGGERED"      # For stops
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    EXPIRED = "EXPIRED"
    CANCELED = "CANCELED"

class TimeInForce(Enum):
    DAY = "DAY"
    IOC = "IOC"  # Immediate-Or-Cancel (for Market/MOC)

@dataclass
class Order:
    """Base order class."""
    symbol: str
    side: OrderSide
    qty: int
    order_type: OrderType
    strategy_ts: datetime
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: OrderState = OrderState.SUBMITTED
    remaining: int = 0
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    tif: TimeInForce = TimeInForce.DAY
    
    def __post_init__(self):
        if self.remaining == 0:
            self.remaining = self.qty
```

##### 2.2 Position Tracker (`src/ledger/positions.py`)

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict

@dataclass
class Position:
    """Position in a single symbol."""
    symbol: str
    qty: int = 0
    avg_price: Decimal = Decimal("0")
    
    @property
    def market_value(self) -> Decimal:
        """Current market value (requires price from ledger)."""
        # Price will be injected at EOD from last bar close
        return Decimal("0")
    
    def update_on_fill(self, fill_qty: int, fill_price: Decimal):
        """Update position on fill using average cost."""
        if fill_qty == 0:
            return
        
        new_qty = self.qty + fill_qty
        
        if new_qty == 0:
            # Closed position
            self.qty = 0
            self.avg_price = Decimal("0")
        elif self.qty * fill_qty >= 0:
            # Adding to position (same side)
            total_cost = (self.qty * self.avg_price) + (fill_qty * fill_price)
            self.avg_price = total_cost / new_qty
            self.qty = new_qty
        else:
            # Reducing/flipping position
            if abs(new_qty) < abs(self.qty):
                # Reducing: keep avg_price
                self.qty = new_qty
            else:
                # Flipping: new avg_price
                self.qty = new_qty
                self.avg_price = fill_price

class PositionTracker:
    """Track positions across all symbols."""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol (creates if not exists)."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update(self, symbol: str, fill_qty: int, fill_price: Decimal):
        """Update position on fill."""
        pos = self.get_position(symbol)
        pos.update_on_fill(fill_qty, fill_price)
    
    def get_net_position(self, symbol: str) -> int:
        """Get net position quantity."""
        return self.get_position(symbol).qty
    
    def is_short(self, symbol: str) -> bool:
        """Check if position is short."""
        return self.get_net_position(symbol) < 0
```

##### 2.3 Cash Ledger (`src/ledger/cash.py`)

```python
from decimal import Decimal
from typing import List, Tuple
from datetime import datetime

class CashLedger:
    """Track cash balance and transactions."""
    
    def __init__(self, initial_capital: Decimal):
        self.balance = initial_capital
        self.transactions: List[Tuple[datetime, str, Decimal]] = []
    
    def debit(self, amount: Decimal, reason: str, ts: datetime):
        """Debit (reduce) cash."""
        self.balance -= amount
        self.transactions.append((ts, f"DEBIT: {reason}", -amount))
    
    def credit(self, amount: Decimal, reason: str, ts: datetime):
        """Credit (increase) cash."""
        self.balance += amount
        self.transactions.append((ts, f"CREDIT: {reason}", amount))
    
    def get_balance(self) -> Decimal:
        """Get current cash balance."""
        return self.balance
```

##### 2.4 Order Manager (`src/execution/order_manager.py`)

```python
from typing import List, Dict
from src.models.order import Order, OrderState, OrderSide
from src.config.engine_config import TradingConfig
import logging

logger = logging.getLogger(__name__)

class OrderManager:
    """Manage order lifecycle and validation."""
    
    def __init__(self, trading_config: TradingConfig, position_tracker):
        self.trading_config = trading_config
        self.position_tracker = position_tracker
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.rejected_orders: List[Tuple[Order, str]] = []
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit order for execution.
        
        Returns True if accepted, False if rejected.
        """
        # Validate qty
        if order.qty <= 0:
            reason = f"Invalid quantity: {order.qty}"
            self.rejected_orders.append((order, reason))
            logger.warning(f"Order rejected: {reason}")
            return False
        
        # Check short selling
        if not self.trading_config.allow_short:
            current_pos = self.position_tracker.get_net_position(order.symbol)
            if order.side == OrderSide.SELL:
                # Check if sell would create short
                if current_pos - order.qty < 0:
                    reason = f"Short selling disabled; position={current_pos}, sell qty={order.qty}"
                    self.rejected_orders.append((order, reason))
                    logger.warning(f"Order rejected: {reason}")
                    return False
        
        # Accept order
        self.pending_orders[order.order_id] = order
        logger.info(f"Order submitted: {order.order_id} {order.side.value} {order.qty} {order.symbol}")
        return True
    
    def get_pending_orders(self, symbol: str = None) -> List[Order]:
        """Get pending orders, optionally filtered by symbol."""
        if symbol:
            return [o for o in self.pending_orders.values() if o.symbol == symbol]
        return list(self.pending_orders.values())
    
    def mark_filled(self, order_id: str):
        """Move order from pending to filled."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.state = OrderState.FILLED
            order.remaining = 0
            self.filled_orders.append(order)
    
    def mark_expired(self, order_id: str):
        """Mark order as expired."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.state = OrderState.EXPIRED
```

#### Tests (`tests/stage2/`)

```python
# tests/stage2/test_order_model.py
def test_order_creation_assigns_uuid():
    """Order should get unique UUID."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        strategy_ts=datetime.now(pytz.UTC),
    )
    assert order.order_id is not None
    assert len(order.order_id) == 36  # UUID format

def test_order_state_transitions():
    """Order state should transition correctly."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.LIMIT,
        strategy_ts=datetime.now(pytz.UTC),
        limit_price=Decimal("150.00"),
    )
    assert order.state == OrderState.SUBMITTED
    
    order.state = OrderState.FILLED
    assert order.state == OrderState.FILLED

# tests/stage2/test_positions.py
def test_position_update_on_buy():
    """Position should update correctly on buy fill."""
    pos = Position(symbol="AAPL")
    pos.update_on_fill(100, Decimal("150.00"))
    
    assert pos.qty == 100
    assert pos.avg_price == Decimal("150.00")

def test_position_update_multiple_buys():
    """Position should calculate average cost on multiple buys."""
    pos = Position(symbol="AAPL")
    pos.update_on_fill(100, Decimal("150.00"))  # 100 @ 150
    pos.update_on_fill(50, Decimal("160.00"))   # 50 @ 160
    
    # Average: (100*150 + 50*160) / 150 = 153.33
    assert pos.qty == 150
    assert abs(pos.avg_price - Decimal("153.3333")) < Decimal("0.01")

def test_position_tracks_short():
    """Position should track short correctly."""
    tracker = PositionTracker()
    tracker.update("AAPL", -100, Decimal("150.00"))
    
    assert tracker.get_net_position("AAPL") == -100
    assert tracker.is_short("AAPL") is True

# tests/stage2/test_cash_ledger.py
def test_cash_ledger_initial_balance():
    """Ledger should track initial capital."""
    ledger = CashLedger(initial_capital=Decimal("100000"))
    assert ledger.get_balance() == Decimal("100000")

def test_cash_debit_credit():
    """Ledger should handle debits and credits."""
    ledger = CashLedger(initial_capital=Decimal("100000"))
    
    ledger.debit(Decimal("15000"), "Buy 100 AAPL", datetime.now(pytz.UTC))
    assert ledger.get_balance() == Decimal("85000")
    
    ledger.credit(Decimal("16000"), "Sell 100 AAPL", datetime.now(pytz.UTC))
    assert ledger.get_balance() == Decimal("101000")

# tests/stage2/test_order_manager.py
def test_order_manager_accepts_valid_order():
    """Manager should accept valid orders."""
    config = TradingConfig(allow_short=False)
    tracker = PositionTracker()
    manager = OrderManager(config, tracker)
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        strategy_ts=datetime.now(pytz.UTC),
    )
    
    assert manager.submit_order(order) is True
    assert len(manager.pending_orders) == 1

def test_order_manager_rejects_short_when_disabled():
    """Manager should reject short when disabled."""
    config = TradingConfig(allow_short=False)
    tracker = PositionTracker()
    manager = OrderManager(config, tracker)
    
    # No position, try to sell (would create short)
    order = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        qty=100,
        order_type=OrderType.MARKET,
        strategy_ts=datetime.now(pytz.UTC),
    )
    
    assert manager.submit_order(order) is False
    assert len(manager.rejected_orders) == 1
```

#### Acceptance Criteria

- ✅ Orders created with all required fields
- ✅ Position tracker updates on fills (buy/sell)
- ✅ Cash ledger tracks debits/credits
- ✅ Order manager validates short selling
- ✅ All Stage 2 tests pass

______________________________________________________________________

### **Stage 3: Execution Engine — Market & MOC**

**Timeline:** Days 5-7\
**Branch:** `stage-3-market-moc`

#### Deliverables

##### 3.1 Fill Model (`src/models/fill.py`)

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from src.models.order import OrderSide

@dataclass
class Fill:
    """Represents a fill (partial or complete)."""
    fill_id: str
    order_id: str
    execution_ts: datetime
    symbol: str
    side: OrderSide
    qty: int
    price: Decimal
    slip_bps: int
    fees: Decimal
    participation: Decimal  # Fraction of bar volume
    partial_index: int  # 0 for full fill, >0 for partials
```

##### 3.2 Cost Model (`src/config/engine_config.py`)

```python
from pydantic import BaseModel, Field
from decimal import Decimal

class CostConfig(BaseModel):
    """Commission and fee configuration."""
    per_share: Decimal = Field(default=Decimal("0.0005"))
    ticket_min: Decimal = Field(default=Decimal("1.00"))

class FillConfig(BaseModel):
    """Fill policy configuration."""
    limit_mode: str = Field(default="conservative")
    stop_mode: str = Field(default="conservative")
    moc_slip_bps: int = Field(default=5)
    slippage_bps: int = Field(default=0)
    max_participation: Decimal = Field(default=Decimal("0.10"))
    queue_bars: int = Field(default=3)
    allow_high_participation: bool = Field(default=False)

class TradingConfig(BaseModel):
    """Trading rules configuration."""
    allow_short: bool = Field(default=False)
    borrow_rate_annual: Decimal = Field(default=Decimal("0.03"))

class EngineConfig(BaseModel):
    """Complete engine configuration."""
    fills: FillConfig = Field(default_factory=FillConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
```

##### 3.3 Fill Policy (`src/execution/fill_policy.py`)

```python
from decimal import Decimal
from typing import Optional
from src.models.bar import Bar
from src.models.order import Order, OrderType, OrderSide
from src.models.fill import Fill
from src.config.engine_config import FillConfig, CostConfig
import uuid

class FillPolicy:
    """Implements fill rules (conservative by default)."""
    
    def __init__(self, fill_config: FillConfig, cost_config: CostConfig):
        self.fill_config = fill_config
        self.cost_config = cost_config
    
    def evaluate_market(self, order: Order, bar: Bar, next_bar_open: Optional[Decimal]) -> Optional[Fill]:
        """
        Market orders fill at next bar open.
        
        Returns None if next_bar_open not available.
        """
        if next_bar_open is None:
            return None
        
        fill_price = next_bar_open
        fees = self._calculate_fees(order.qty)
        
        return Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            execution_ts=bar.ts,  # Strategy ts; fill executes at next bar
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            slip_bps=self.fill_config.slippage_bps,
            fees=fees,
            participation=Decimal("0"),  # Market doesn't check participation
            partial_index=0,
        )
    
    def evaluate_moc(self, order: Order, bar: Bar) -> Fill:
        """
        Market-On-Close fills at bar close with slippage.
        
        Buy: close + slip_bps
        Sell: close - slip_bps
        """
        close_price = bar.close
        slip_bps = self.fill_config.moc_slip_bps
        
        if order.side == OrderSide.BUY:
            fill_price = close_price * (Decimal("1") + Decimal(slip_bps) / Decimal("10000"))
        else:
            fill_price = close_price * (Decimal("1") - Decimal(slip_bps) / Decimal("10000"))
        
        fees = self._calculate_fees(order.qty)
        
        return Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            execution_ts=bar.ts,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            slip_bps=slip_bps,
            fees=fees,
            participation=Decimal("0"),
            partial_index=0,
        )
    
    def _calculate_fees(self, qty: int) -> Decimal:
        """Calculate commission: max(per_share * qty, ticket_min)."""
        per_share_fee = self.cost_config.per_share * qty
        return max(per_share_fee, self.cost_config.ticket_min)
```

##### 3.4 Execution Engine (`src/execution/engine.py`)

```python
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
import logging

from src.models.bar import Bar
from src.models.order import Order, OrderType, OrderState
from src.models.fill import Fill
from src.execution.order_manager import OrderManager
from src.execution.fill_policy import FillPolicy
from src.ledger.positions import PositionTracker
from src.ledger.cash import CashLedger
from src.config.engine_config import EngineConfig

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """Main execution engine coordinating order processing."""
    
    def __init__(
        self,
        config: EngineConfig,
        initial_capital: Decimal,
    ):
        self.config = config
        self.position_tracker = PositionTracker()
        self.cash_ledger = CashLedger(initial_capital)
        self.order_manager = OrderManager(config.trading, self.position_tracker)
        self.fill_policy = FillPolicy(config.fills, config.costs)
        
        self.fills: List[Fill] = []
        self.bar_cache: Dict[str, Bar] = {}  # symbol -> last bar
        self.next_open_cache: Dict[str, Decimal] = {}  # symbol -> next open
    
    def submit_order(self, order: Order) -> bool:
        """Submit order to order manager."""
        return self.order_manager.submit_order(order)
    
    def process_bar(self, bar: Bar, is_close_only: bool = False):
        """
        Process a single bar through the event loop.
        
        Steps:
        1. Cache bar for next-bar-open lookups
        2. Evaluate intrabar (limit/stop) - skip if close_only
        3. End-of-bar: MOC fills, schedule Market for next
        4. Apply fills to ledger
        5. EOD accruals (borrow, dividends)
        """
        self.bar_cache[bar.symbol] = bar
        
        # Update next_open_cache from previous bar
        if bar.symbol in self.next_open_cache:
            del self.next_open_cache[bar.symbol]
        
        # Schedule next open for Market orders
        self.next_open_cache[bar.symbol] = bar.open  # Will be used by next bar
        
        # End-of-bar: process MOC
        self._process_moc_orders(bar)
        
        # Process scheduled Market orders (from previous bar)
        self._process_market_orders(bar)
        
        # Apply fills
        for fill in self.fills:
            if fill.execution_ts == bar.ts and fill.symbol == bar.symbol:
                self._apply_fill(fill)
    
    def _process_moc_orders(self, bar: Bar):
        """Process Market-On-Close orders."""
        pending = self.order_manager.get_pending_orders(bar.symbol)
        
        for order in pending:
            if order.order_type == OrderType.MOC:
                fill = self.fill_policy.evaluate_moc(order, bar)
                self.fills.append(fill)
                self.order_manager.mark_filled(order.order_id)
                logger.info(f"MOC fill: {fill.qty} {fill.symbol} @ {fill.price}")
    
    def _process_market_orders(self, bar: Bar):
        """Process Market orders scheduled from previous bar."""
        pending = self.order_manager.get_pending_orders(bar.symbol)
        
        for order in pending:
            if order.order_type == OrderType.MARKET:
                # Use current bar open as "next open"
                fill = self.fill_policy.evaluate_market(order, bar, bar.open)
                if fill:
                    self.fills.append(fill)
                    self.order_manager.mark_filled(order.order_id)
                    logger.info(f"Market fill: {fill.qty} {fill.symbol} @ {fill.price}")
    
    def _apply_fill(self, fill: Fill):
        """Apply fill to positions and cash."""
        # Update position
        fill_qty = fill.qty if fill.side == OrderSide.BUY else -fill.qty
        self.position_tracker.update(fill.symbol, fill_qty, fill.price)
        
        # Update cash
        gross_amount = fill.price * fill.qty
        total_cost = gross_amount + fill.fees
        
        if fill.side == OrderSide.BUY:
            self.cash_ledger.debit(total_cost, f"Buy {fill.qty} {fill.symbol}", fill.execution_ts)
        else:
            self.cash_ledger.credit(gross_amount - fill.fees, f"Sell {fill.qty} {fill.symbol}", fill.execution_ts)
```

#### Tests (`tests/stage3/`)

```python
# tests/stage3/test_fill_policy.py
def test_moc_buy_adds_slippage():
    """MOC buy should add slip_bps to close."""
    config = FillConfig(moc_slip_bps=5)
    cost_config = CostConfig()
    policy = FillPolicy(config, cost_config)
    
    bar = Bar(
        ts=datetime.now(pytz.UTC),
        symbol="AAPL",
        open=Decimal("150"),
        high=Decimal("152"),
        low=Decimal("149"),
        close=Decimal("151"),
        volume=1000000,
    )
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MOC,
        strategy_ts=bar.ts,
    )
    
    fill = policy.evaluate_moc(order, bar)
    
    # Expected: 151 * (1 + 0.0005) = 151.0755
    expected = Decimal("151") * Decimal("1.0005")
    assert abs(fill.price - expected) < Decimal("0.01")
    assert fill.slip_bps == 5

def test_commission_applies_ticket_min():
    """Commission should enforce ticket minimum."""
    config = FillConfig()
    cost_config = CostConfig(per_share=Decimal("0.0005"), ticket_min=Decimal("1.00"))
    policy = FillPolicy(config, cost_config)
    
    # Small order: 10 shares * 0.0005 = 0.005 < 1.00 min
    fees = policy._calculate_fees(10)
    assert fees == Decimal("1.00")
    
    # Large order: 10000 shares * 0.0005 = 5.00 > 1.00 min
    fees = policy._calculate_fees(10000)
    assert fees == Decimal("5.00")

# tests/stage3/test_execution_engine.py
def test_engine_processes_moc_order():
    """Engine should fill MOC orders at bar close."""
    config = EngineConfig()
    engine = ExecutionEngine(config, initial_capital=Decimal("100000"))
    
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150"),
        high=Decimal("152"),
        low=Decimal("149"),
        close=Decimal("151"),
        volume=1000000,
    )
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MOC,
        strategy_ts=bar.ts,
    )
    
    engine.submit_order(order)
    engine.process_bar(bar)
    
    # Should have 1 fill
    assert len(engine.fills) == 1
    fill = engine.fills[0]
    assert fill.qty == 100
    assert fill.symbol == "AAPL"
    
    # Position should be updated
    assert engine.position_tracker.get_net_position("AAPL") == 100
    
    # Cash should be debited
    assert engine.cash_ledger.get_balance() < Decimal("100000")

def test_engine_schedules_market_for_next_bar():
    """Engine should fill Market orders at next bar open."""
    config = EngineConfig()
    engine = ExecutionEngine(config, initial_capital=Decimal("100000"))
    
    bar1 = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150"),
        high=Decimal("152"),
        low=Decimal("149"),
        close=Decimal("151"),
        volume=1000000,
    )
    
    bar2 = Bar(
        ts=datetime(2023, 1, 2, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("152"),  # Next open
        high=Decimal("154"),
        low=Decimal("151"),
        close=Decimal("153"),
        volume=1000000,
    )
    
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        strategy_ts=bar1.ts,
    )
    
    engine.submit_order(order)
    engine.process_bar(bar1)  # Market not filled yet
    
    assert len(engine.fills) == 0
    
    engine.process_bar(bar2)  # Fill at bar2 open
    
    assert len(engine.fills) == 1
    fill = engine.fills[0]
    assert fill.price == Decimal("152")  # bar2.open
```

#### Acceptance Criteria

- ✅ Market orders fill at next bar open
- ✅ MOC orders fill at close with slippage
- ✅ Commissions applied with ticket minimum
- ✅ Positions updated on fills
- ✅ Cash debited/credited correctly
- ✅ All Stage 3 tests pass

______________________________________________________________________

### **Stage 4-6: Continuation**

Due to length constraints, I'll summarize the remaining stages:

**Stage 4:** Limit & Stop orders with conservative rules and close-only bar handling\
**Stage 5:** Volume participation caps, partial fills, residual queuing\
**Stage 6:** Short dividends, borrow costs, output writers (CSV/JSON)

______________________________________________________________________

## 🎨 Golden Baseline Generation

### Approach

Create standalone scripts in `scripts/goldens/` to run reference strategies:

#### 1. Buy-and-Hold Strategy (`scripts/goldens/buy_and_hold.py`)

```python
"""
Generate golden baseline for Buy-and-Hold strategy.

Strategy:
- Submit Market buy order on first bar
- Hold until last bar
- Calculate final NAV

Run on AAPL, MSFT, AMZN (2019-2023)
"""
# ... implementation
```

#### 2. SMA Cross Strategy (`scripts/goldens/sma_cross.py`)

```python
"""
Generate golden baseline for SMA Cross strategy.

Strategy:
- Calculate 50-day and 200-day SMAs
- Buy when fast > slow (golden cross)
- Sell when fast < slow (death cross)

Run on MSFT (2019-2023)
"""
# ... implementation
```

### Golden File Format

```json
{
  "strategy": "buy_and_hold",
  "symbol": "AAPL",
  "version": "1.0",
  "fixture_hash": "abc123...",
  "schema_version": "algoseek-parquet-v1.0",
  "date_range": ["2019-01-02", "2023-12-29"],
  "config": { ...  },
  "results": {
    "initial_capital": 100000.00,
    "final_nav": 125432.15,
    "total_return": 0.254321,
    "num_trades": 1,
    "fills": [ ... ],
    "daily_nav": { ... }
  }
}
```

### Validation

- Run golden generation scripts manually
- Review results together
- Commit golden files to `tests/goldens/fixtures/`
- Create test that loads goldens and validates determinism

______________________________________________________________________

## 📦 Dependencies to Add

```toml
# Add to pyproject.toml dependencies
dependencies = [
    "duckdb>=1.4.0",
    "pandas>=2.3.2",
    "pyarrow>=21.0.0",
    "click>=8.0.0",
    "pydantic>=2.11.9",
    "pyyaml>=6.0",      # NEW: For YAML config loading
    "pytz>=2024.1",     # NEW: For timezone handling
]
```

______________________________________________________________________

## 🚦 Success Metrics

### Per-Stage

- All tests pass (`make test`)
- Code quality passes (`make qa`: ruff, isort, mypy)
- Pre-commit hooks pass

### Overall Phase 1

- Load 3,774 bars from fixture
- Execute Market, MOC, Limit, Stop orders
- Handle volume participation and partials
- Track positions and cash with Decimal precision
- Generate 3 golden baselines (Buy-Hold AAPL/MSFT/AMZN)
- All 6 stages merged to main

______________________________________________________________________

## ✅ Next Steps

1. **Confirm column mapping** - ✅ Done (see schema analysis)
1. **Review implementation plan** - ✅ This document
1. **Start Stage 1** - Await your approval to proceed
1. **Golden baseline review** - Schedule after Stage 6 complete

______________________________________________________________________

**Ready to start Stage 1?** 🚀
