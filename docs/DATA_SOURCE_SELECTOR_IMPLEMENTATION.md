# Data Source Selector - Clean Implementation Plan

**Status:** Phase 1 - Not Yet In Production **Approach:** Clean implementation (no backward compatibility needed) **Timeline:** 2-3 weeks

______________________________________________________________________

## 🎯 Goal

Replace the simple `source_tag` string with a structured `DataSourceSelector` system that scales to multiple providers, asset classes, and data types.

______________________________________________________________________

## 📋 Sprint 1: Core Implementation (Week 1-2)

### Day 1-2: Create Core Classes

**File:** `src/qtrader/config/data_source_selector.py`

```python
"""Data source selector for flexible provider/asset matching."""

from dataclasses import dataclass, field
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
    OHLCV = "ohlcv"
    TRADES = "trades"
    QUOTES = "quotes"
    GREEKS = "greeks"
    FUNDAMENTALS = "fundamentals"


@dataclass
class DataSourceSelector:
    """
    Structured selector for data sources.

    Specify only the criteria that matter for your use case.
    Resolver will find the best matching source.

    Examples:
        # Specific provider
        DataSourceSelector(provider="schwab", asset_class=AssetClass.EQUITY)

        # Any provider with fallback
        DataSourceSelector(
            asset_class=AssetClass.EQUITY,
            frequency="1d",
            fallback_providers=["schwab", "algoseek"]
        )

        # CME futures
        DataSourceSelector(
            asset_class=AssetClass.FUTURES,
            exchange="CME"
        )
    """

    # Core selection criteria
    provider: Optional[str] = None
    asset_class: Optional[AssetClass] = None
    data_type: DataType = DataType.OHLCV
    frequency: Optional[str] = None

    # Optional refinements
    exchange: Optional[str] = None
    region: Optional[str] = None
    adjustment_mode: Optional[str] = None

    # Fallback providers (try in order)
    fallback_providers: list[str] = field(default_factory=list)

    def matches(self, source_config: dict) -> bool:
        """Check if source config matches this selector's criteria."""
        # Only check specified fields (None = don't care)
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
        """Generate human-readable tag for logging."""
        parts = []
        if self.provider:
            parts.append(self.provider)
        if self.asset_class:
            parts.append(self.asset_class.value)
        if self.data_type != DataType.OHLCV:
            parts.append(self.data_type.value)
        if self.frequency:
            parts.append(self.frequency)
        if self.exchange:
            parts.append(self.exchange)
        return "-".join(parts) if parts else "any"
```

**Tests:** `tests/unit/config/test_data_source_selector.py`

### Day 3: Update DataConfig

**File:** `src/qtrader/config/data_config.py`

```python
from qtrader.config.data_source_selector import DataSourceSelector

class DataConfig(BaseModel):
    """Data loading and processing configuration."""

    mode: str = Field(default="adjusted")
    frequency: str = Field(default="1d")
    timezone: str = Field(default="America/New_York")

    # NEW: Structured selector (required)
    source_selector: DataSourceSelector = Field(
        description="Data source selector - provider, asset class, etc."
    )

    bar_schema: BarSchemaConfig = Field(...)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @validator('source_selector')
    def validate_selector(cls, v):
        """Ensure at least one criterion specified."""
        if not any([v.provider, v.asset_class, v.exchange, v.frequency]):
            raise ValueError(
                "DataSourceSelector must specify at least one criterion "
                "(provider, asset_class, exchange, or frequency)"
            )
        return v
```

**Changes:**

- ✅ Remove `source_tag` field (no backward compat)
- ✅ Add `source_selector` as required field
- ✅ Add validation

**Tests:** Update `tests/unit/config/test_data_config.py`

### Day 4-5: Update Resolver

**File:** `src/qtrader/adapters/resolver.py`

```python
def resolve_by_selector(
    self,
    selector: DataSourceSelector,
    instrument: Instrument,
) -> Any:
    """
    Find best matching data source for selector.

    Args:
        selector: Structured selector with criteria
        instrument: Instrument to load data for

    Returns:
        Configured adapter instance

    Raises:
        ValueError: If no matching source found
    """
    # Find matching sources
    matches = []
    for source_name, source_config in self.sources["data_sources"].items():
        if selector.matches(source_config):
            matches.append((source_name, source_config))

    if not matches:
        available = list(self.sources["data_sources"].keys())
        raise ValueError(
            f"No data source matches selector: {selector.to_tag()}\n"
            f"Available sources: {available}\n"
            f"Selector criteria: provider={selector.provider}, "
            f"asset_class={selector.asset_class}, frequency={selector.frequency}"
        )

    # Use first match (or implement priority logic later)
    if len(matches) > 1:
        logger.info(
            "resolver.multiple_matches",
            selector=selector.to_tag(),
            matches=[m[0] for m in matches],
            selected=matches[0][0],
        )

    source_name, source_config = matches[0]

    # Try primary source
    try:
        return self._create_adapter(source_name, source_config, instrument)
    except Exception as e:
        logger.warning(
            "resolver.primary_failed",
            source=source_name,
            error=str(e),
        )

        # Try fallback providers
        for fallback_provider in selector.fallback_providers:
            try:
                fallback_selector = DataSourceSelector(
                    provider=fallback_provider,
                    asset_class=selector.asset_class,
                    data_type=selector.data_type,
                    frequency=selector.frequency,
                    exchange=selector.exchange,
                    region=selector.region,
                )
                logger.info(
                    "resolver.trying_fallback",
                    fallback=fallback_provider,
                )
                return self.resolve_by_selector(fallback_selector, instrument)
            except Exception:
                continue

        # No fallbacks worked, raise original error
        raise

def resolve(self, instrument: Instrument) -> Any:
    """Legacy method for backward compat with Instrument.data_source enum."""
    # Convert enum to selector
    source_map = {
        DataSource.ALGOSEEK: DataSourceSelector(provider="algoseek"),
        DataSource.SCHWAB: DataSourceSelector(provider="schwab"),
    }
    selector = source_map.get(instrument.data_source)
    if not selector:
        raise ValueError(f"Unknown data source enum: {instrument.data_source}")
    return self.resolve_by_selector(selector, instrument)
```

**Tests:** Update `tests/unit/adapters/test_resolver.py`

### Day 6: Update data_sources.yaml

**File:** `config/data_sources.yaml`

```yaml
# Data Sources Configuration
# Each source specifies provider, asset class, and capabilities

data_sources:

  # Algoseek - US Equities Daily (Unadjusted)
  algoseek-us-equity-daily:
    # Source metadata
    provider: algoseek
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: unadjusted

    # Adapter configuration
    adapter: algoseekOHLC
    root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
    mode: standard_adjusted
    path_template: "{root_path}/SecId={secid}/*.parquet"
    symbol_map: "data/equity_security_master_sample.csv"

  # Schwab - US Equities Daily (Split-Adjusted)
  schwab-us-equity-daily:
    # Source metadata
    provider: schwab
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: adjusted  # Note: Schwab only provides split-adjusted

    # OAuth configuration
    client_id: "${SCHWAB_API_KEY}"
    client_secret: "${SCHWAB_API_SECRET}"
    redirect_uri: "${SCHWAB_REDIRECT_URI:-https://127.0.0.1:8182}"
    token_cache_path: null
    manual_mode: false

    # Rate limiting
    rate_limit: 120
    rate_period: 60
```

### Day 7-8: Update Tests

**Update all tests to use new selector:**

1. `tests/unit/config/test_data_config.py` - DataConfig validation
1. `tests/unit/adapters/test_resolver.py` - Resolver matching logic
1. `tests/unit/data/test_loader.py` - DataLoader with selector
1. `tests/unit/services/test_data_service.py` - DataService with selector
1. `tests/integration/services/test_data_service_integration.py` - End-to-end

### Day 9: Update Examples

**File:** `examples/services/data_service_example.py`

```python
from qtrader.config import BarSchemaConfig, DataConfig, DataSourceSelector, AssetClass

# Configure data selector
selector = DataSourceSelector(
    provider="schwab",
    asset_class=AssetClass.EQUITY,
    frequency="1d",
)

config = DataConfig(
    mode="adjusted",
    frequency="1d",
    timezone="America/New_York",
    source_selector=selector,
    bar_schema=bar_schema,
)

service = DataService(config)
```

### Day 10: QA & Documentation

- ✅ Run full test suite
- ✅ Update README.md
- ✅ Update architecture docs
- ✅ Code review

______________________________________________________________________

## 📋 Sprint 2: Polish & Features (Week 3)

### Features to Add

1. **Factory Methods**

```python
# In DataConfig
@classmethod
def for_provider(
    cls,
    provider: str,
    asset_class: AssetClass,
    frequency: str = "1d",
    mode: str = "adjusted",
) -> "DataConfig":
    """Create config for specific provider (convenience)."""
    selector = DataSourceSelector(
        provider=provider,
        asset_class=asset_class,
        frequency=frequency,
    )
    return cls(
        mode=mode,
        frequency=frequency,
        source_selector=selector,
        bar_schema=BarSchemaConfig.default(),
    )
```

2. **Source Discovery API**

```python
# In DataSourceResolver
def list_available_sources(
    self,
    asset_class: Optional[AssetClass] = None,
    frequency: Optional[str] = None,
) -> list[dict]:
    """List available sources matching criteria."""
    sources = []
    for name, config in self.sources["data_sources"].items():
        if asset_class and config.get("asset_class") != asset_class.value:
            continue
        if frequency and config.get("frequency") != frequency:
            continue
        sources.append({
            "name": name,
            "provider": config.get("provider"),
            "asset_class": config.get("asset_class"),
            "frequency": config.get("frequency"),
            "region": config.get("region"),
        })
    return sources
```

3. **Validation at Load Time**

```python
# In DataSourceResolver.__init__
def _validate_sources(self):
    """Validate source configurations."""
    required_fields = ["provider", "asset_class", "data_type"]
    for name, config in self.sources["data_sources"].items():
        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Source '{name}' missing required field: {field}"
                )
```

______________________________________________________________________

## 📋 Sprint 3: Future Assets (As Needed)

### When Adding Futures

1. Add to `config/data_sources.yaml`:

```yaml
cme-futures-daily:
    provider: cmeDataMine
    asset_class: futures
    data_type: ohlcv
    frequency: 1d
    exchange: CME
    # ... adapter config
```

2. Create example: `examples/data_sources/futures_example.py`
1. Add tests

### When Adding Crypto

Similar process for crypto, options, etc.

______________________________________________________________________

## ✅ Success Criteria

- [ ] All 241 existing tests passing
- [ ] New tests for selector matching (>10 test cases)
- [ ] `data_service_example.py` uses new selector
- [ ] Documentation updated
- [ ] QA checks passing (ruff, isort, mypy)
- [ ] No `source_tag` references in code

______________________________________________________________________

## 🚀 Quick Start (After Implementation)

```python
from qtrader.config import DataConfig, DataSourceSelector, AssetClass
from qtrader.services import DataService

# Simple: specific provider
selector = DataSourceSelector(
    provider="schwab",
    asset_class=AssetClass.EQUITY,
)

config = DataConfig.for_provider("schwab", AssetClass.EQUITY)
service = DataService(config)

# Advanced: any provider with fallback
selector = DataSourceSelector(
    asset_class=AssetClass.EQUITY,
    frequency="1d",
    fallback_providers=["schwab", "algoseek"],
)
```

______________________________________________________________________

## 📝 Notes

- **No Backward Compatibility**: Clean break from `source_tag`
- **Type Safety**: Full mypy compliance
- **Extensible**: Easy to add new asset classes
- **Flexible**: Match on any combination of criteria
- **Resilient**: Fallback provider support
