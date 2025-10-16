# Data Source Selection - Future-Proof Design

## Problem Statement

Current approach uses a simple `source_tag` string (e.g., `"algoseek-adjusted"`, `"schwab-adjusted"`), which doesn't scale well for multiple providers, asset classes, and data types.

### Current Datasets

1. **Algoseek**

   - Provider: algoseek
   - Asset Class: US Equities
   - Frequency: 1d
   - Mode: unadjusted (raw)

1. **Schwab**

   - Provider: schwab
   - Asset Class: US Equities
   - Frequency: 1d
   - Mode: adjusted (split-adjusted only)

### Future Requirements

Need to support:

- **Multiple Providers**: algoseek, schwab, IQFeed, Polygon, Bloomberg, etc.
- **Multiple Asset Classes**: equities, futures, options, crypto, forex
- **Multiple Frequencies**: 1m, 5m, 15m, 1h, 1d, 1w
- **Multiple Data Types**: OHLCV, trades, quotes, greeks, fundamentals
- **Specific Exchanges**: CME, ICE, NASDAQ, NYSE, Binance, etc.

### Challenges with Current Approach

```python
# ❌ Current: Simple string tag
source_tag = "algoseek-adjusted"  # Hard to parse, limited info
source_tag = "provider-x-cme-futures-1h-ohlc"  # Gets unwieldy

# ❌ How to express:
# - "Give me daily OHLC for US equities from ANY provider"
# - "Prefer Schwab, fallback to Algoseek"
# - "CME futures from provider that supports 1h frequency"
```

______________________________________________________________________

## Proposed Solution: Structured Data Source Selector

### 1. New DataSourceSelector Class

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AssetClass(Enum):
    """Asset class categories."""
    EQUITY = "equity"
    FUTURES = "futures"
    OPTIONS = "options"
    CRYPTO = "crypto"
    FOREX = "forex"
    FIXED_INCOME = "fixed_income"


class DataType(Enum):
    """Type of market data."""
    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TRADES = "trades"  # Tick-by-tick trades
    QUOTES = "quotes"  # Bid/ask quotes
    GREEKS = "greeks"  # Options greeks
    FUNDAMENTALS = "fundamentals"  # Company fundamentals
    DEPTH = "depth"  # Order book depth


@dataclass
class DataSourceSelector:
    """
    Structured selector for data sources with flexible matching.

    All fields are optional - specify only what matters for your use case.
    Resolver will find best matching source based on provided criteria.
    """

    # Core attributes
    provider: Optional[str] = None  # "algoseek", "schwab", "iqfeed", etc.
    asset_class: Optional[AssetClass] = None  # equity, futures, etc.
    data_type: DataType = DataType.OHLCV  # Default to OHLCV
    frequency: Optional[str] = None  # "1m", "5m", "1h", "1d", etc.

    # Optional refinements
    exchange: Optional[str] = None  # "NYSE", "NASDAQ", "CME", "ICE", etc.
    region: Optional[str] = None  # "US", "EU", "APAC", etc.
    adjustment_mode: Optional[str] = None  # "adjusted", "unadjusted", "split_adjusted"

    # Fallback behavior
    fallback_providers: list[str] = None  # Try in order if primary fails
    require_adjustment: bool = False  # Must support unadjusted data

    def matches(self, source_config: dict) -> bool:
        """
        Check if source config matches selector criteria.

        Args:
            source_config: Data source configuration from YAML

        Returns:
            True if all specified criteria match
        """
        # Only check fields that are specified (not None)
        if self.provider and source_config.get("provider") != self.provider:
            return False

        if self.asset_class and source_config.get("asset_class") != self.asset_class.value:
            return False

        if self.data_type and source_config.get("data_type") != self.data_type.value:
            return False

        if self.frequency and source_config.get("frequency") != self.frequency:
            return False

        if self.exchange and source_config.get("exchange") != self.exchange:
            return False

        if self.region and source_config.get("region") != self.region:
            return False

        if self.adjustment_mode and source_config.get("adjustment_mode") != self.adjustment_mode:
            return False

        return True

    def to_tag(self) -> str:
        """Generate a human-readable tag for logging."""
        parts = []
        if self.provider:
            parts.append(self.provider)
        if self.asset_class:
            parts.append(self.asset_class.value)
        if self.data_type != DataType.OHLCV:  # Skip default
            parts.append(self.data_type.value)
        if self.frequency:
            parts.append(self.frequency)
        if self.exchange:
            parts.append(self.exchange)
        return "-".join(parts) if parts else "any"
```

### 2. Updated DataConfig (Clean - No Legacy)

```python
class DataConfig(BaseModel):
    """Data loading and processing configuration."""

    # Mode for internal processing
    mode: str = Field(
        default="adjusted",
        description="Data adjustment mode (adjusted|unadjusted|total_return)",
    )

    # Frequency and timezone
    frequency: str = Field(default="1d", description="Bar frequency (1m|5m|15m|1h|1d)")
    timezone: str = Field(default="America/New_York", description="Timezone for timestamps")

    # Structured data source selection
    source_selector: DataSourceSelector = Field(
        description="Structured data source selector - specifies provider, asset class, etc.",
    )

    # Schema mappings
    bar_schema: BarSchemaConfig = Field(
        description="Column mapping for bar data",
    )

    # Validation
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="OHLC validation rules",
    )

    strict_frequency: bool = Field(default=True, description="Raise on frequency mismatch")
    decimals: dict[str, int] = Field(default={"price": 4, "cash": 4}, description="Decimal precision")

    @validator('source_selector')
    def validate_selector(cls, v):
        """Ensure at least one selection criterion is specified."""
        if not any([v.provider, v.asset_class, v.data_type, v.frequency, v.exchange]):
            raise ValueError("DataSourceSelector must specify at least one criterion")
        return v
```

### 3. Updated data_sources.yaml

```yaml
# Data Sources Configuration
# Each source specifies provider, asset class, and capabilities

data_sources:

  # Algoseek - US Equities Daily (Unadjusted)
  algoseek-us-equity-daily:
    provider: algoseek
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: unadjusted

    # Adapter config
    adapter: algoseekOHLC
    root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
    path_template: "{root_path}/SecId={secid}/*.parquet"
    symbol_map: "data/equity_security_master_sample.csv"

  # Schwab - US Equities Daily (Split-Adjusted Only)
  schwab-us-equity-daily:
    provider: schwab
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: adjusted  # Note: Schwab only provides split-adjusted

    # OAuth credentials
    client_id: "${SCHWAB_API_KEY}"
    client_secret: "${SCHWAB_API_SECRET}"
    redirect_uri: "${SCHWAB_REDIRECT_URI:-https://127.0.0.1:8182}"
    token_cache_path: null
    manual_mode: false

  # Example: IQFeed - US Equities 1-minute
  iqfeed-us-equity-1m:
    provider: iqfeed
    asset_class: equity
    data_type: ohlcv
    frequency: 1m
    region: US

    api_key: "${IQFEED_API_KEY}"
    api_secret: "${IQFEED_API_SECRET}"

  # Example: CME - Futures Daily
  cme-futures-daily:
    provider: cmeDataMine
    asset_class: futures
    data_type: ohlcv
    frequency: 1d
    exchange: CME

    api_token: "${CME_API_TOKEN}"
    s3_bucket: "${CME_S3_BUCKET}"

  # Example: Binance - Crypto 5-minute
  binance-crypto-5m:
    provider: binance
    asset_class: crypto
    data_type: ohlcv
    frequency: 5m

    api_key: "${BINANCE_API_KEY}"
    api_secret: "${BINANCE_API_SECRET}"
```

### 4. Resolver with Flexible Matching

```python
class DataSourceResolver:
    """Resolves DataSourceSelector to appropriate adapter."""

    def resolve_by_selector(
        self,
        selector: DataSourceSelector,
        instrument: Instrument,
    ) -> Any:
        """
        Find best matching data source for selector.

        Args:
            selector: Structured selector with desired attributes
            instrument: Instrument to load data for

        Returns:
            Configured adapter instance

        Raises:
            ValueError: If no matching source found
        """
        # Find all matching sources
        matches = []
        for source_name, source_config in self.sources["data_sources"].items():
            if selector.matches(source_config):
                matches.append((source_name, source_config))

        if not matches:
            raise ValueError(
                f"No data source matches selector: {selector.to_tag()}\n"
                f"Available sources: {list(self.sources['data_sources'].keys())}"
            )

        # If multiple matches, use first (or implement priority logic)
        if len(matches) > 1:
            logger.info(
                "resolver.multiple_matches",
                selector=selector.to_tag(),
                matches=[m[0] for m in matches],
                selected=matches[0][0],
            )

        source_name, source_config = matches[0]

        # Try primary provider
        try:
            return self._create_adapter(source_name, source_config, instrument)
        except Exception as e:
            # Try fallback providers if specified
            if selector.fallback_providers:
                for fallback in selector.fallback_providers:
                    try:
                        fallback_selector = DataSourceSelector(
                            provider=fallback,
                            asset_class=selector.asset_class,
                            data_type=selector.data_type,
                            frequency=selector.frequency,
                        )
                        return self.resolve_by_selector(fallback_selector, instrument)
                    except Exception:
                        continue
            raise

    def resolve(self, instrument: Instrument) -> Any:
        """Legacy method - uses instrument.data_source enum."""
        # Convert enum to selector for backward compatibility
        source_mapping = {
            DataSource.ALGOSEEK: DataSourceSelector(provider="algoseek"),
            DataSource.SCHWAB: DataSourceSelector(provider="schwab"),
        }
        selector = source_mapping.get(instrument.data_source)
        if not selector:
            raise ValueError(f"Unknown data source: {instrument.data_source}")
        return self.resolve_by_selector(selector, instrument)
```

______________________________________________________________________

## Usage Examples

### Example 1: Specific Provider

```python
from qtrader.config import DataConfig, DataSourceSelector, AssetClass

# Use specific provider
selector = DataSourceSelector(
    provider="schwab",
    asset_class=AssetClass.EQUITY,
    frequency="1d",
)

config = DataConfig(
    mode="adjusted",
    frequency="1d",
    source_selector=selector,
    bar_schema=bar_schema,
)

service = DataService(config)
```

### Example 2: Any Provider with Fallback

```python
# Prefer Schwab, fallback to Algoseek
selector = DataSourceSelector(
    asset_class=AssetClass.EQUITY,
    frequency="1d",
    region="US",
    fallback_providers=["schwab", "algoseek"],
)

config = DataConfig(
    mode="adjusted",
    source_selector=selector,
    bar_schema=bar_schema,
)
```

### Example 3: CME Futures

```python
# CME futures, any provider that supports them
selector = DataSourceSelector(
    asset_class=AssetClass.FUTURES,
    exchange="CME",
    frequency="1d",
)

config = DataConfig(
    mode="unadjusted",  # Futures typically don't need adjustment
    source_selector=selector,
    bar_schema=bar_schema,
)
```

### Example 4: Crypto High-Frequency

```python
# Crypto 5-minute bars
selector = DataSourceSelector(
    asset_class=AssetClass.CRYPTO,
    frequency="5m",
    provider="binance",  # Specify if you have preference
)

config = DataConfig(
    mode="unadjusted",  # Crypto doesn't have corporate actions
    source_selector=selector,
    bar_schema=bar_schema,
)
```

### Example 5: Simple Default (Most Common Case)

```python
# Most common case: just specify provider and asset class
# Uses sensible defaults for everything else
selector = DataSourceSelector(
    provider="schwab",
    asset_class=AssetClass.EQUITY,
)

config = DataConfig(
    mode="adjusted",
    frequency="1d",
    source_selector=selector,
    bar_schema=bar_schema,
)

# Or even simpler with a factory helper:
config = DataConfig.for_provider(
    provider="schwab",
    asset_class=AssetClass.EQUITY,
)
```

______________________________________________________________________

## Implementation Strategy (Clean Slate - No Legacy Support)

### Phase 1: Core Implementation ✅ RECOMMENDED

Since the project is not yet in production, we can skip backward compatibility and implement the clean design from the start.

**Week 1: Core Classes**

1. ✅ Create `AssetClass` enum
1. ✅ Create `DataType` enum
1. ✅ Create `DataSourceSelector` dataclass with matching logic
1. ✅ Update `DataConfig` to use `source_selector` (remove `source_tag`)
1. ✅ Add tests for selector matching

**Week 2: Resolver Integration** 6. ✅ Update `DataSourceResolver.resolve_by_selector()` 7. ✅ Update `data_sources.yaml` with structured metadata 8. ✅ Add fallback provider logic 9. ✅ Update all unit tests

**Week 3: Examples & Documentation** 10. ✅ Update `data_service_example.py` with new selector 11. ✅ Add example for futures (when available) 12. ✅ Add example for crypto (when available) 13. ✅ Update README and documentation

**Week 4: Polish** 14. ✅ Add source discovery API (`list_available_sources()`) 15. ✅ Add validation at config load time 16. ✅ Performance testing with multiple sources 17. ✅ Integration testing end-to-end

### Phase 2: Expand Asset Classes (As Needed)

**When Adding Futures:**

1. Add futures data source to `data_sources.yaml`
1. Create example: `examples/data_sources/futures_example.py`
1. Add futures-specific tests

**When Adding Crypto:**

1. Add crypto data source to `data_sources.yaml`
1. Create example: `examples/data_sources/crypto_example.py`
1. Add crypto-specific tests

**When Adding Options:**

1. Add options data source to `data_sources.yaml`
1. Create example: `examples/data_sources/options_example.py`
1. Add greeks support to `DataType` enum

### Phase 3: Advanced Features (Future)

**Performance Optimization:**

- Cache source selection results
- Parallel data loading from multiple sources
- Smart fallback with health checks

**Enhanced Discovery:**

- API to query source capabilities
- Automatic provider selection based on availability
- Cost optimization (use cheapest provider that meets criteria)

**Monitoring:**

- Track provider performance/reliability
- Alert on provider failures
- Automatic failover based on SLA

______________________________________________________________________

## Benefits

### ✅ **Explicit and Type-Safe**

- Clear what each field means
- IDE autocomplete for enum values
- Pydantic validation

### ✅ **Flexible Matching**

- Specify only what matters
- "Give me daily equities from ANY provider"
- "Give me CME futures, prefer provider X"

### ✅ **Scales to New Asset Classes**

- Add futures: Just set `asset_class=AssetClass.FUTURES`
- Add crypto: Just set `asset_class=AssetClass.CRYPTO`
- Easy to extend with new providers

### ✅ **Fallback Support**

- Try multiple providers in priority order
- Resilient to provider outages
- Easy to switch providers

### ✅ **Clean Design**

- No legacy cruft
- No deprecated fields
- Simple, obvious API

### ✅ **Clear Configuration**

- YAML structure mirrors selector
- Easy to see what each source provides
- Self-documenting

______________________________________________________________________

## Implementation Priority (Clean Implementation)

### 🟢 **Sprint 1: Core Implementation** (Week 1-2)

**Goal:** Replace `source_tag` with `DataSourceSelector` completely

1. ✅ Create enums (`AssetClass`, `DataType`)
1. ✅ Create `DataSourceSelector` dataclass
1. ✅ Update `DataConfig` - replace `source_tag` with `source_selector`
1. ✅ Update `DataSourceResolver` to use selector
1. ✅ Update `data_sources.yaml` with structured metadata
1. ✅ Update all tests (no backward compat needed!)
1. ✅ Update `data_service_example.py`

**Deliverable:** Working system with new selector, all tests passing

### 🟡 **Sprint 2: Polish & Features** (Week 3)

**Goal:** Add convenience features and validation

8. ✅ Add fallback provider logic
1. ✅ Add selector validation at config load time
1. ✅ Add factory methods for common cases
1. ✅ Add source discovery API
1. ✅ Update documentation

**Deliverable:** Production-ready with great developer experience

### 🔴 **Sprint 3: Future Assets** (As Needed)

**Goal:** Add support for new asset classes when ready

13. ⏳ Add futures support (when data source available)
01. ⏳ Add crypto support (when data source available)
01. ⏳ Add options support (when data source available)
01. ⏳ Add forex support (when data source available)

**Deliverable:** Multi-asset support ready for production strategies

______________________________________________________________________

## Questions to Consider

1. **Priority/Ranking**: If multiple sources match, how to choose? (First match, explicit priority, performance metrics?)

1. **Capabilities**: Should sources declare what they DON'T support? (e.g., Schwab can't provide unadjusted)

1. **Caching**: Should selector be part of cache key? (Different cache for different frequencies)

1. **Validation**: Should we validate selector against available sources at config time?

1. **Discovery**: API to list available sources matching criteria?

______________________________________________________________________

## Recommendation

**Start with Phase 1**: Add `DataSourceSelector` alongside existing `source_tag`, implement basic matching, update one example. This gives you the flexibility you need without breaking anything.

Then iterate based on actual usage patterns as you add more providers and asset classes.
