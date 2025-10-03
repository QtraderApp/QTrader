# Equities Backtesting Engine — Specification (Phase 1, **v1.0**)

**Owner:** Javier **Readers:** Quant researchers, software engineers, data engineers **Version:** 1.0 — Approved baseline for implementation **Scope:** U.S. equities, bar‑based backtests (1m–1d). Deterministic by design.

______________________________________________________________________

## 1. Purpose & Non‑Goals

### 1.1 Purpose

Build a **deterministic**, **auditable**, and **extensible** equities backtesting engine with realistic (but configurable) execution modeling suitable for professional quants.

### 1.2 Non‑Goals (Phase 1)

- No live trading adapters; no broker emulation beyond simple fills.
- No PIT reference data enforcement (user responsible for PIT universes; engine logs assumptions).
- No FX/multi‑currency. USD only.
- No advanced risk/tearsheet metrics (Sharpe/Sortino/etc.) — P2.

______________________________________________________________________

## 2. Architecture Overview

### 2.1 Ports & Adapters

- **DataPort** → adapters (Algoseek Parquet, CSV, future vendors) normalize to canonical **Bar**.
- **ExecPort** → execution policies (fill rules, participation, slippage).
- **CostPort** → commissions/fees (per‑share + ticket min).
- **RiskPort** (minimal P1) → equity/notional checks.

### 2.2 Determinism

- Single‑threaded event loop per backtest run.
- Lexicographic tie‑break on equal timestamps across symbols.
- Fixed RNG seed for any stochastic components (not used in P1 by default).

### 2.3 Vendor Integration Model & Bar Contract (Authoritative)

**Design goal:** Accept **any vendor schema** → transform to a **canonical Bar** used end‑to‑end across the trade lifecycle (signals → orders → fills → accounting).

**Bar is vendor-agnostic, asset-agnostic, and frequency-agnostic.** Works with equities, futures, crypto, forex at any timeframe. Vendor-specific fields (adjustments, bid/ask, etc.) are stored separately in `AdjustmentEvent`.

**Bar Contract (OHLCV only):**

```python
class Bar(NamedTuple):
    """
    Canonical OHLCV bar - the ONLY contract consumed by execution engine.

    This is vendor-agnostic, asset-agnostic, and frequency-agnostic.
    Works with equities, futures, crypto, forex at any timeframe.
    """
    ts: datetime         # timezone-aware per data.timezone
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
```

**AdjustmentEvent (Separate Storage):**

```python
class AdjustmentEvent(NamedTuple):
    """
    Corporate action metadata for analysis and validation.

    NOT used by execution engine. Stored for:
    - Audit trail (data provenance)
    - Performance attribution (dividend-adjusted returns)
    - Data validation (detect missing adjustments)
    """
    ts: datetime                      # Event timestamp (ex-date)
    symbol: str
    event_type: str                   # CashDiv, Split, StockDiv, SpinOff
    px_factor: Decimal                # Cumulative price adjustment factor
    vol_factor: Decimal               # Cumulative volume adjustment factor
    metadata: Dict[str, Any]          # Vendor-specific details (amount, ratio, etc.)
```

**DataMode Enum:**

```python
class DataMode(Enum):
    """Declares whether dataset is adjusted or unadjusted."""
    ADJUSTED = "adjusted"              # Total-return adjusted (splits + dividends)
    UNADJUSTED = "unadjusted"          # Raw prices, no adjustments
    SPLIT_ADJUSTED = "split_adjusted"  # Split-adjusted only, no dividends
```

**Adapter interface:**

```python
class DataAdapter(Protocol):
    def can_read(self, source: URI) -> bool: ...
    def schema_version(self) -> str: ...
    def get_data_mode(self) -> DataMode: ...
    def read_bars(self, source: URI, config: DataConfig) -> Iterable[Bar]: ...
    def read_adjustments(self, source: URI, config: DataConfig) -> Iterable[AdjustmentEvent]: ...
```

**Pipeline:** `Read vendor data → Map schema → Emit Bar + AdjustmentEvent`.

**Schema mapping (config-driven):**

- `bar_schema` maps vendor columns to OHLCV fields (ts, symbol, open, high, low, close, volume)
- `adjustment_schema` (optional) maps vendor columns to adjustment fields
- No code changes needed for new vendors - just config

**Extensibility rules:**

- New vendor = new `DataAdapter` implementation + config mapping; engine code unchanged
- The **Bar** type is the **only** price/volume contract consumed by Exec/Cost/Risk
- **AdjustmentEvent** is optional metadata for audit/analysis, not execution
- Adapter declares `DataMode` to indicate if data is adjusted or unadjusted
- Optionally persist a **Standard Bar Extract** to disk (parquet) to speed reruns

**Validation:**

- Strict type/required checks at adapter boundary
- Frequency/monotonic checks after normalization
- OHLC relationship validation (high ≥ max(o,c), low ≤ min(o,c))

**Versioning:**

- Adapters advertise `schema_version()`; runs persist this in `run.json` for reproducibility

**Naming convention:**

- Adapters named: `{Vendor}{AssetClass}{Frequency}{DataType}Adapter`
- Example: `AlgoseekUSEquityDailyOHLCAdapter`, `IQFeedUSEquityMinuteOHLCAdapter`

### 2.4 Multi-Dataset Support

**Design goal:** Strategies can access multiple datasets (primary OHLCV + auxiliary alternative data, factors, or cross-validation sources).

**Primary vs Auxiliary:**

- **Primary dataset:** Drives the event loop; `on_bar()` called for each bar
- **Auxiliary datasets:** Queried on-demand by strategy via `ctx.get_data()`

**Configuration:**

```yaml
# Single dataset (backward compatible)
data:
  source: "data/algoseek/"
  adapter: "algoseek_us_equity_daily_ohlc"
  mode: adjusted
  frequency: 1d

# Multi-dataset (new in Phase 1A)
primary:
  name: "algoseek"
  source: "data/algoseek/"
  adapter: "algoseek_us_equity_daily_ohlc"
  mode: adjusted
  frequency: 1d
  bar_schema:
    ts: "TradeDate"
    symbol: "Ticker"
    open: "Open"
    high: "High"
    low: "Low"
    close: "Close"
    volume: "MarketHoursVolume"

auxiliary:
  - name: "news"
    source: "data/news_sentiment.csv"
    adapter: "csv"
    frequency: 1d
    schema:
      ts: "date"
      symbol: "ticker"
      fields:
        sentiment_score: float
        article_count: int

  - name: "factors"
    source: "data/factors.parquet"
    adapter: "parquet"
    frequency: 1d
    schema:
      ts: "date"
      symbol: "ticker"
      fields:
        value_z: float
        momentum_z: float

alignment:
  strategy: forward_fill    # forward_fill | drop | error
  max_lookback_days: 5      # Max forward-fill window
  require_primary: true     # Must have primary data
```

**Context API for multi-dataset:**

```python
# In strategy on_bar() method
def on_bar(self, bar: Bar, ctx: Context):
    # bar is from PRIMARY dataset
    price = bar.close

    # Query AUXILIARY datasets
    sentiment = ctx.get_data("news", bar.symbol, bar.ts, "sentiment_score")
    value_z = ctx.get_data("factors", bar.symbol, bar.ts, "value_z")

    # Trading logic
    if sentiment > 0.7 and value_z > 1.5:
        ctx.buy_market(bar.symbol, 100)
```

**Alignment strategies:**

- `forward_fill`: Fill missing data from last available value (up to N days back)
- `drop`: Skip bars without data in all datasets
- `error`: Raise exception on missing data

**Phase 1A scope:**

- Multiple datasets at same frequency (e.g., all daily)
- Primary + auxiliary configuration
- Alignment strategies (forward_fill, drop, error)

**Phase 1B scope (future):**

- Mixed frequency support (daily primary + intraday auxiliary)
- Time window queries: `ctx.get_bars("iqfeed_1m", symbol, start, end)`

______________________________________________________________________

## 3. Dataset Alignment (Algoseek **Standard Adjusted** OHLC)

**Assumption:** Vendor bars are **total‑return adjusted** — both **dividends and splits** embedded via cumulative factors. Implications:

- **Long dividends:** **Do not** post separate long cash dividends; already in price path.
- **Short dividends:** **Do** post **negative cash** on **ex‑date** for symbols held short when `AdjustmentEvent.event_type` indicates a cash dividend. Dividend amount derived from adjustment metadata.
- **Splits / scrip / rights:** Already reflected in adjusted prices. P1 does not mutate share counts; lot/share CA mechanics are P2.

**How adjustment events are used:**

- **Bar (OHLCV):** Used by execution engine for trading decisions and fills
- **AdjustmentEvent:** Used by ledger for short dividend debits; stored for audit trail and performance attribution
- Adapters emit both streams separately: `read_bars()` and `read_adjustments()`

### 3.1 Canonical `Bar` (JSON representation)

```json
{
  "ts": "YYYY‑MM‑DD[THH:MM]",
  "symbol": "AAPL",
  "open": 197.1200,
  "high": 198.5000,
  "low": 195.8000,
  "close": 197.9100,
  "volume": 32456789
}
```

**Note:** Adjustment metadata (dividends, splits) stored separately in `AdjustmentEvent`:

```json
{
  "ts": "2023-02-10",
  "symbol": "AAPL",
  "event_type": "CashDiv",
  "px_factor": 1.2345,
  "vol_factor": 0.8123,
  "metadata": {
    "dividend_amount": 0.24,
    "currency": "USD"
  }
}
```

**Precision:** Prices serialized at `data.decimals.price` (default 4). Indicator math uses float64; ledger uses Decimal.

### 3.2 Data Config

**Single dataset (simple):**

```yaml
data:
  source: "data/algoseek/"
  adapter: "algoseek_us_equity_daily_ohlc"
  mode: adjusted                   # adjusted | unadjusted | split_adjusted
  frequency: 1d                    # 1m|5m|15m|1h|1d
  timezone: America/New_York
  strict_frequency: true           # raise if mismatch
  decimals: {price: 4, cash: 4}

  # Schema mapping (vendor columns → canonical Bar fields)
  bar_schema:
    ts: "TradeDate"
    symbol: "Ticker"
    open: "Open"
    high: "High"
    low: "Low"
    close: "Close"
    volume: "MarketHoursVolume"

  # Adjustment metadata (optional)
  adjustment_schema:
    ts: "TradeDate"
    symbol: "Ticker"
    event_type: "AdjustmentReason"
    px_factor: "CumulativePriceFactor"
    vol_factor: "CumulativeVolumeFactor"

  validation:
    epsilon: 0.0
    ohlc_policy: strict_raise      # strict_raise | warn_skip_bar | warn_use_close_only
```

**Multi-dataset (advanced):**

See §2.4 for multi-dataset configuration with primary + auxiliary datasets.

### 3.3 Integrity Checks

- Validate monotonic timestamps per symbol.
- Median delta must match `frequency` when `strict_frequency=true`.
- If `mode=standard_adjusted`, **disable long dividends** and **enable short dividend debits**.

### 3.4 Data Validation Policies (dataset‑specific)

Raw vendor bars can be wrong (e.g., `high < max(open,close)` or `low > min(open,close)`). Validation and fallback are **dataset‑specific** and controlled via YAML.

**Validation rules:**

- `high ≥ max(open, close) − epsilon`
- `low ≤ min(open, close) + epsilon`
- `low ≤ high + epsilon`
- `volume ≥ 0`

`epsilon` is a small tolerance (default `0.0` for daily; vendors sometimes require `1e‑6` on intraday).

**Policy (per‑dataset):**

```yaml
data:
  validation:
    epsilon: 0.0
    ohlc_policy: strict_raise   # strict_raise | warn_skip_bar | warn_use_close_only
    close_only_fields: [close]  # when warn_use_close_only, only these fields are trusted
```

- **`strict_raise`**: raise an error and stop run on first malformed bar.
- **`warn_skip_bar`**: log a warning, **skip the bar** (no fills, orders remain pending).
- **`warn_use_close_only`** (aka *close‑only/brown‑bar mode*): log a warning, **trust only `close` (and fields in `close_only_fields`)** and treat the bar as **ineligible for intrabar touch logic** (see §4.1/§4.3 interaction). Execution that requires `high/low` is **disabled for that bar**; queued orders remain pending to next bar. Market/MOC still function.

**Run metadata:**

- Engine records counts per policy: `{malformed_bars: N, skipped: X, close_only: Y}` in `run.json` and emits first/last 10 offending `(date,symbol)` pairs.

______________________________________________________________________

### 3.5 Reference Dataset (Fixtures)

**Purpose:** Single source of truth for tests, examples, and golden files.

- **Location:** `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample` (project‑relative `data/` folder).

- **Format:** **Parquet**.

- **Partitioning:** by **`SecId`**. Path template: `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/SecId=<SecId>/*.parquet`.

- **Universe:** `MSFT`, `AMZN`, `AAPL`.

- **Date range:** **2019‑01‑01** to **2023‑12‑31** (inclusive).

- **Frequency:** `1d` (daily) for P1 fixtures.

- **Column map (example):** `{ ts: trade_date, symbol: ticker, open: open, high: high, low: low, close: close, volume: market_hours_volume }`.

- **Usage rules:**

  - All **unit/integration tests**, **reference strategies** (e.g., Buy‑and‑Hold, SMA Cross), and **golden results** MUST run against this fixture by default.
  - CI loads fixture the same way as users: through the **DataAdapter → Bar** normalization pipeline.
  - Any change to fixture or adapter **invalidates goldens**; bump golden version and store alongside `run.json`.

______________________________________________________________________

## 4. Orders & Execution

### 4.1 Order Types

- **Market** — Default for close‑based signals is **Next‑Bar‑Open** (no same‑bar close fills). Intrabar market fills allowed only when strategy is intrabar and the bar passes OHLC validation.
- **Market‑On‑Close (MOC)** — Executes on current bar’s close/auction with configurable auction slippage (bps). Allowed in all policies.
- **Limit** — Touch rules configurable; conservative by default. **Disabled for the current bar** if `ohlc_policy=warn_use_close_only` is applied because high/low cannot be trusted; order remains pending.
- **Stop (Stop→Market)** — Trigger then market with conservative rule by default. **Disabled for the current bar** if `ohlc_policy=warn_use_close_only` is applied; order remains pending.

### 4.2 Time‑In‑Force (TIF)

- Market/MOC ignore TIF (IOC by nature).
- Limit/Stop default **DAY** (expire at end of bar/day). GTC is P2.

### 4.3 Fill Policy (Configurable)

```yaml
fills:
  limit_mode: conservative        # conservative|optimistic
  stop_mode: conservative         # conservative|optimistic
  moc: { slip_bps: 5 }            # default 5 bps auction slippage
  slippage_bps: 0                 # per‑fill generic slippage
  max_participation: 0.10         # ≤10% of bar volume (guardrail warns above 0.20)
  queue_bars: 3                   # carry unfilled qty up to N bars
  allow_high_participation: false # require explicit override if >0.20
```

**Conservative rules:**

- **Limit Buy:** if `low ≤ limit` then fill at `min(limit, close)`; else no fill.
- **Limit Sell:** if `high ≥ limit` then fill at `max(limit, close)`; else no fill.
- **Stop Buy:** if `high ≥ stop` then fill at `max(stop, close)` ± slippage.
- **Stop Sell:** if `low ≤ stop` then fill at `min(stop, close)` ± slippage.
- **MOC:** fill at close price ± `moc.slip_bps` (bp add for buys, subtract for sells).

**Governance:** The team **pins conservative** as default. Switching `limit_mode`/`stop_mode` to `optimistic` requires code review sign‑off and a change log entry in the repository.

### 4.4 Volume Participation & Partial Fills (ENFORCED)

- Max shares fill per bar per order side: `cap = max_participation × bar.volume`.
- If requested qty > `cap`, engine **partially fills** up to `cap` and **queues residual** forward for up to `queue_bars` bars; residual expires afterward.
- Each partial is a distinct fill slice with its own costs/slippage.

### 4.5 Order Lifecycle & States

`SUBMITTED → (TRIGGERED for stops) → PARTIALLY_FILLED* → FILLED | EXPIRED | CANCELED`

______________________________________________________________________

## 5. Shorting, Borrow, and Dividends

```yaml
trading:
  allow_short: false              # default false
  borrow_rate_annual: 0.03        # flat annual; accrues EOD on short MV
```

- If `allow_short=false`, orders that would create net short are rejected.

- If `allow_short=true`:

  - **Borrow cost accrual:** each EOD: `cash -= |short_market_value| × (borrow_rate_annual/252)`.
  - **Short dividends:** on **ex‑date** when `adj_reason` indicates cash dividend and net short at bar close (EOD), post `cash -= |shares| × dividend_per_share`.
  - **Long dividends:** none (embedded in adjusted prices).

**Dividend source:** For P1, `dividend_per_share` is derived from vendor CA metadata or implied from price factors when available; if missing, engine logs a warning and skips cash posting (prices still reflect the event).

______________________________________________________________________

## 6. Calendar & Session Semantics

- No external session calendar in P1. “**Next‑open**” = next available bar in the dataset.
- Holidays/halts/early closes are implicit in the bar series; pending DAY orders roll to next bar.
- MOC executes on the current bar close (including early close bars).

______________________________________________________________________

## 7. Event Loop (Canonical)

```python
# Phase 1: Initialization
strategy.on_init(ctx)                    # register custom indicators

# Phase 2: Warmup (if indicators.warmup=true)
if warmup_enabled:
  for bar in warmup_bars:
    emit(bar)
    # Process bar for indicator computation only
    # Do NOT call strategy.on_bar()
    # Portfolio remains at initial state

# Phase 3: Strategy Start
strategy.on_start(ctx)                   # after warmup completes

# Phase 4: Trading Loop
for bar in dataset:
  emit(bar)
  strategy.on_bar(bar)                   # submit orders (strategy_ts = bar.ts)
  exec.evaluate_intrabar(bar)            # limit/stop touches; apply participation; partials
  exec.end_of_bar(bar)                   # MOC on current; schedule Market for next open
  ledger.apply_fills_and_costs()
  ledger.eod_accruals()                  # borrow, short dividends if ex‑date
  outputs.snapshot_if_eod()
  logs.write()

# Phase 5: Finalization
strategy.on_end(ctx)
```

**Timestamps:** each fill slice records `strategy_ts` (submission) and `execution_ts` (fill). Engine clock = bar timestamps; no sub‑bar clock in P1.

**Warmup behavior:** When `indicators.warmup=true`, bars in the warmup period are processed to compute indicator values but `strategy.on_bar()` is NOT called. Trading begins after warmup completes.

______________________________________________________________________

## 8. Configuration (Full Example)

```yaml
seed: 42

data:
  mode: standard_adjusted
  frequency: 1d
  timezone: America/New_York
  strict_frequency: true
  decimals: {price: 4, cash: 4}
  source_tag: algoseek-standard-adjusted
  validation:
    epsilon: 0.0
    ohlc_policy: warn_use_close_only   # strict_raise | warn_skip_bar | warn_use_close_only
    close_only_fields: [close]

fills:
  limit_mode: conservative
  stop_mode: conservative
  moc: {slip_bps: 5}
  slippage_bps: 0
  max_participation: 0.10
  queue_bars: 3
  allow_high_participation: false

trading:
  allow_short: true
  borrow_rate_annual: 0.03

costs:
  per_share: 0.0005
  ticket_min: 1.00

indicators:
  warmup: false           # Enable automatic indicator warmup (default: false)
  warmup_bars: null       # Auto-detect max lookback, or specify explicit count
```

______________________________________________________________________

## 9. Costs & Commissions

- **Per‑share:** `costs.per_share` × filled shares.
- **Ticket minimum:** apply once per order per bar slice (`fills.csv` shows enforced minimum flag).
- Costs are deducted from cash at the time of each fill.

### 9.1 Ledger Computation Order (authoritative)

For every fill slice: **(1) start from execution price**, **(2) apply slippage (bps) to price**, **(3) compute gross cash impact**, **(4) apply fees/commissions**, **(5) update positions and cash**, **(6) quantize ledger values to Decimal with configured `data.decimals`**.

______________________________________________________________________

## 10. Output Artifacts & Schemas

### 10.1 `performance.json`

```json
{
  "run_id": "uuid",
  "start_date": "YYYY‑MM‑DD",
  "end_date": "YYYY‑MM‑DD",
  "ann_return": 0.1123,
  "ann_vol": 0.1820,
  "daily_returns": {"2024‑01‑02": 0.0012, "2024‑01‑03": -0.0007}
}
```

### 10.2 `positions_daily.csv`

`date,symbol,qty,avg_price,market_value,cash,nav,exposure_long,exposure_short`

### 10.3 `orders.csv`

`order_id,strategy_ts,symbol,side,qty,type,limit,stop,tif,state,remaining`

### 10.4 `fills.csv`

`fill_id,order_id,execution_ts,symbol,side,qty,price,slip_bps,fees,participation,partial_index`

### 10.5 `run.json`

- config snapshot, code version/hash, RNG seed, `data.source_tag`, universe text, warnings.

### 10.6 `signals.jsonl` (optional)

One JSON object per line containing strategy debug data.

______________________________________________________________________

## 11. Error Handling & Validation

- **Data frequency:** raise when `strict_frequency=true` and median delta mismatch.

- **NaN/negative prices or volumes:** raise with symbol/date detail.

- **Malformed OHLC bars:** apply `data.validation.ohlc_policy`:

  - `strict_raise` → error out with first offending `(date,symbol)`.
  - `warn_skip_bar` → warn and skip; count in `run.json`.
  - `warn_use_close_only` → warn and mark bar *close‑only*; disallow limit/stop evaluation on that bar and record in `run.json`.

- **High participation guardrail:** if `fills.max_participation > 0.20` and `fills.allow_high_participation=false`, **warn and clamp to 0.20** for the run; record in `run.json`.

- **Order rejections:** invalid qty/side, shorting disabled, TIF violations, etc.

- **Adjustment sanity:** warn if `adj_reason` missing on large discontinuities; configurable `adjustments.enforce: warn|raise`.

______________________________________________________________________

## 12. Testing Strategy (P1)

### 12.1 Fixture Definition (authoritative)

- Dataset: **Parquet**, partitioned by **`SecId`**, under `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample`.
- Universe: **MSFT, AMZN, AAPL**.
- Date range: **2019‑01‑01 → 2023‑12‑31**.
- Frequency: **1d**.

### 12.2 Unit Tests

- Limit/Stop conservative vs optimistic fills (all four cases) using fixture bars.

- Market (next‑open) vs MOC with `moc.slip_bps` on fixture dates with high/low bounds.

- Participation cap with residual queue over multiple bars within fixture.

- Short dividend debit on known ex‑dates present in adjusted series; borrow accrual daily.

- Commission model with ticket minimum.

- Strict frequency and bad data guards using synthetic corrupted copies of fixture.

- **Malformed bar policies:**

  - Inject bars with `high < max(open,close)` and verify `strict_raise` aborts.
  - Inject bars with `low > min(open,close)` and verify `warn_skip_bar` keeps orders pending.
  - Inject either case and verify `warn_use_close_only` allows **Market/MOC** but **prevents limit/stop** fills for the offending bar.

### 12.3 Golden Files (Determinism)

- **Buy‑and‑Hold** each of MSFT, AMZN, AAPL → canonical NAV paths (no costs and with costs).
- **SMA Cross** on MSFT → canonical fills & PnL.
- Goldens are stored under `tests/goldens/<strategy>/<version>/` with `run.json` including adapter `schema_version()` and fixture hash.

### 12.4 CI Fixture Integrity

- Compute and store a **checksum/hash** of the fixture directories (e.g., per‑partition hash + global Merkle root) in repo (`tests/fixtures/manifest.json`) and verify at CI start.
- If mismatch: fail fast with guidance; include expected vs observed hash in logs.
- Persist the used fixture hash into `run.json` for auditability.

______________________________________________________________________

## 13. Performance & Numerics

- Indicator math in float64 for speed; ledger and serialized prices in Decimal (quantized to `data.decimals`).
- Target: 10 years × 3k symbols × 1d bars under 60s on a modern CPU (non‑parallel). Provide `numerics.mode: {float_indicators|full_decimal}` toggle (default `float_indicators`).

______________________________________________________________________

## 13A. Indicators & Technical Analysis

### 13A.1 Overview

QTrader provides a comprehensive **Indicators Framework** for computing technical analysis indicators from bar data. The framework supports:

- **Built-in indicators**: SMA, EMA, Bollinger Bands, ATR, RSI, MACD
- **Custom indicators**: User-defined indicators inheriting from `Indicator` base class
- **Indicator helpers**: 13 utility functions for crossover, threshold, divergence, histogram, and trend detection
- **Efficient caching**: Incremental updates, O(1) per bar for rolling indicators
- **Float64 math**: Per §13, indicators use float64 for performance (ledger remains Decimal)

### 13A.2 Base Indicator Class

All indicators (built-in and custom) inherit from `Indicator[T]`:

```python
from qtrader.api.indicators import Indicator

class MyIndicator(Indicator[float]):
    """Custom indicator implementation."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period

    def compute(self, symbol: str, ctx: Context) -> float | None:
        """
        Compute indicator value for current bar.

        Returns None if insufficient data.
        """
        bars = ctx.get_bar_history(symbol, self.period)
        if len(bars) < self.period:
            return None

        # Your indicator logic here
        ...
```

**Lifecycle:**

1. `__init__()`: Initialize parameters
1. `warmup(symbol, ctx)`: Optional pre-computation (called before on_start)
1. `compute(symbol, ctx)`: Called on each bar to get current value
1. `reset(symbol)`: Clear cached state (called on backtest restart)

### 13A.3 Built-in Indicators API

Access via `ctx.ind.<indicator>()` in strategy methods:

**Simple Moving Average:**

```python
sma_20 = ctx.ind.sma(symbol, period=20, field='close')
```

**Exponential Moving Average:**

```python
ema_12 = ctx.ind.ema(symbol, period=12, field='close')
```

**Bollinger Bands:**

```python
bb = ctx.ind.bollinger_bands(symbol, period=20, num_std=2.0)
if bar.close > bb.upper:
    # Overbought
    ...
```

**Average True Range (Volatility):**

```python
atr_14 = ctx.ind.atr(symbol, period=14)
```

**Relative Strength Index (Momentum):**

```python
rsi_14 = ctx.ind.rsi(symbol, period=14)
if rsi_14 and rsi_14 > 70:
    # Overbought
    ...
```

**MACD:**

```python
macd_line, signal_line, histogram = ctx.ind.macd(symbol, fast=12, slow=26, signal=9)
if histogram > 0:
    # Bullish
    ...
```

### 13A.4 Custom Indicators

**Registration in `on_start()`:**

```python
class MyStrategy(Strategy):
    def on_start(self, ctx: Context):
        """Register custom indicators before trading."""
        ctx.ind.register("momentum", CustomMomentum(period=20))

    def on_bar(self, bar, ctx):
        mom = ctx.ind.get("momentum", bar.symbol)
        if mom and mom > 5.0:
            ctx.buy_market(bar.symbol, 100)
```

**Custom indicator implementation:**

```python
class CustomMomentum(Indicator[float]):
    def __init__(self, period: int = 10):
        super().__init__()
        self.period = period

    def compute(self, symbol: str, ctx) -> float | None:
        bars = ctx.get_bar_history(symbol, self.period + 1)
        if len(bars) < self.period + 1:
            return None

        curr = float(bars[-1].close)
        prev = float(bars[-self.period - 1].close)
        return (curr - prev) / prev * 100.0
```

### 13A.5 Indicator Helper Functions

QTrader provides helper functions for common indicator patterns and signal detection.

**Module:** `qtrader.api.indicator_helpers`

#### Crossover Detection

**Two-Value Crossover (Manual):**

```python
from qtrader.api.indicator_helpers import crossed_above, crossed_below

fast_curr = ctx.ind.sma(bar.symbol, 20)
slow_curr = ctx.ind.sma(bar.symbol, 50)
# ... get previous values ...

if crossed_above(fast_curr, slow_curr, fast_prev, slow_prev):
    ctx.buy_market(bar.symbol, 100)
```

**Context-Tracked Crossover (Recommended):**

```python
# Track indicators
ctx._track_indicator(bar.symbol, 'sma_20', ctx.ind.sma(bar.symbol, 20))
ctx._track_indicator(bar.symbol, 'sma_50', ctx.ind.sma(bar.symbol, 50))

# Check crossover (automatic previous value tracking)
if ctx.crossed_above(bar.symbol, 'sma_20', 'sma_50'):
    ctx.buy_market(bar.symbol, 100)
elif ctx.crossed_below(bar.symbol, 'sma_20', 'sma_50'):
    ctx.sell_market(bar.symbol, 100)
```

#### Threshold Detection

**Crossing Thresholds:**

```python
# Track RSI
rsi = ctx.ind.rsi(bar.symbol, 14)
ctx._track_indicator(bar.symbol, 'rsi_14', rsi)

# RSI crosses above 30 (oversold exit)
if ctx.crossed_above_threshold(bar.symbol, 'rsi_14', 30):
    ctx.buy_market(bar.symbol, 100)

# RSI crosses below 70 (overbought exit)
if ctx.crossed_below_threshold(bar.symbol, 'rsi_14', 70):
    ctx.sell_market(bar.symbol, 100)
```

**Current State Checks:**

```python
from qtrader.api.indicator_helpers import above_threshold, below_threshold, between_thresholds

rsi = ctx.ind.rsi(bar.symbol, 14)

if below_threshold(rsi, 30):
    # RSI is oversold (may have been for multiple bars)
    pass

if above_threshold(rsi, 70):
    # RSI is overbought
    pass

if between_thresholds(rsi, 40, 60):
    # RSI is in neutral zone
    pass
```

#### Available Helper Functions

| Function                       | Purpose                                 | Common Use Case               |
| ------------------------------ | --------------------------------------- | ----------------------------- |
| `crossed_above()`              | Value1 crossed above value2             | Fast SMA > Slow SMA (bullish) |
| `crossed_below()`              | Value1 crossed below value2             | Fast SMA < Slow SMA (bearish) |
| `crossed_above_threshold()`    | Value crossed above threshold           | RSI > 30 (oversold exit)      |
| `crossed_below_threshold()`    | Value crossed below threshold           | RSI < 70 (overbought exit)    |
| `above_threshold()`            | Value currently above threshold         | RSI > 70 (overbought state)   |
| `below_threshold()`            | Value currently below threshold         | RSI < 30 (oversold state)     |
| `between_thresholds()`         | Value in range                          | RSI between 40-60 (neutral)   |
| `divergence_bullish()`         | Price lower low, indicator higher low   | RSI bullish divergence        |
| `divergence_bearish()`         | Price higher high, indicator lower high | RSI bearish divergence        |
| `histogram_flipped_positive()` | Histogram crossed above zero            | MACD bullish momentum         |
| `histogram_flipped_negative()` | Histogram crossed below zero            | MACD bearish momentum         |
| `is_increasing()`              | Indicator trending up N periods         | SMA uptrend confirmation      |
| `is_decreasing()`              | Indicator trending down N periods       | SMA downtrend confirmation    |

### 13A.6 Performance Characteristics

- **Caching**: Indicators cache computed values per (symbol, timestamp)
- **Incremental updates**: Rolling indicators (SMA, EMA) update in O(1) per bar
- **Memory**: Default cache size 1000 values per indicator per symbol
- **Numerics**: Float64 math for speed (ledger/serialized prices remain Decimal)

### 13A.7 Warmup Behavior

By default, indicators return `None` when insufficient data exists (e.g., SMA(20) needs 20 bars).

**Warmup Configuration:**

```yaml
indicators:
  warmup: true              # Enable automatic warmup (default: false)
  warmup_bars: null         # Auto-detect max period, or specify explicit count
```

**Warmup Modes:**

1. **No warmup (default):**

   ```python
   # Strategy must handle None during warmup period
   sma = ctx.ind.sma(bar.symbol, 20)
   if sma is None:
       return  # Skip this bar
   ```

1. **Automatic warmup enabled:**

   ```yaml
   indicators:
     warmup: true
   ```

   - Engine **pre-computes all registered indicators** before calling `on_start()`
   - Warmup period = maximum lookback across all indicators + datasets
   - Example: If SMA(50) is the longest indicator, engine processes 50 bars before `on_start()`
   - Strategy's `on_bar()` is only called after warmup completes
   - **Benefits:** Indicators always return valid values; no None handling needed
   - **Cost:** Delayed strategy start (no fills during warmup period)

1. **Explicit warmup bars:**

   ```yaml
   indicators:
     warmup: true
     warmup_bars: 100  # Process 100 bars before on_start()
   ```

   - Override auto-detection with explicit count
   - Useful when warmup period exceeds indicator lookbacks (e.g., for stable variance estimation)

**Warmup Process (when enabled):**

1. **Detection phase** (before `on_start()`):

   - Engine scans all registered indicators to determine max lookback
   - Example: Strategy uses SMA(20), SMA(50), RSI(14) → max lookback = 50 bars

1. **Warmup phase**:

   - Engine processes `warmup_bars` (or detected max) bars **without calling strategy `on_bar()`**
   - Indicators compute and cache values during this phase
   - Portfolio remains at initial state (no trading)

1. **Trading phase**:

   - After warmup completes, engine calls `strategy.on_start(ctx)`
   - Then begins normal `on_bar()` loop with all indicators ready

**Example with warmup:**

```python
class SMACrossover(Strategy):
    """SMA crossover with automatic warmup."""

    def on_start(self, ctx: Context):
        """Called after warmup completes. Indicators are ready."""
        # No need to register - using built-in indicators
        print(f"Warmup complete. Trading starts at {ctx.current_date}")

    def on_bar(self, bar: Bar, ctx: Context):
        # Indicators ALWAYS return valid values (warmup guarantees this)
        fast = ctx.ind.sma(bar.symbol, 20)
        slow = ctx.ind.sma(bar.symbol, 50)

        # No None checks needed!
        if fast > slow:
            ctx.buy_market(bar.symbol, 100)
        elif fast < slow:
            ctx.sell_market(bar.symbol, 100)
```

**Warmup with custom indicators:**

```python
class MyStrategy(Strategy):
    def on_init(self, ctx: Context):
        """Called BEFORE warmup to register custom indicators."""
        ctx.ind.register("custom_momentum", CustomMomentum(period=30))

    def on_start(self, ctx: Context):
        """Called AFTER warmup completes."""
        # Custom indicators are also warmed up
        pass

    def on_bar(self, bar: Bar, ctx: Context):
        mom = ctx.ind.get("custom_momentum", bar.symbol)
        # mom is always valid (never None) after warmup
```

**Run metadata:**

When warmup is enabled, `run.json` records:

```json
{
  "indicators": {
    "warmup_enabled": true,
    "warmup_bars": 50,
    "warmup_end_date": "2019-02-28",
    "trading_start_date": "2019-03-01"
  }
}
```

**Phase 1 Implementation:**

- ✅ Warmup configuration in YAML
- ✅ Auto-detection of max lookback
- ✅ Explicit `warmup_bars` override
- ✅ `on_init()` lifecycle hook (called before warmup)
- ✅ `on_start()` called after warmup completes
- ✅ Warmup metadata in `run.json`
- ✅ CLI flag: `--warmup` to enable without config change

______________________________________________________________________

## 14. Developer Notes

### 14.1 Order of Operations (Bar)

1. Strategy computes and submits orders.
1. Intrabar evaluation for limit/stop (with participation).
1. End‑bar evaluation: MOC fill; Market scheduled for next bar open.
1. Ledger updates: positions, cash, PnL, costs.
1. EOD accruals: borrow; short dividends on ex‑date.
1. Output snapshots and logs.

### 14.2 Symbol Ordering & Reproducibility

- Per‑bar multi‑symbol processing uses **lexicographic** order for determinism.
- Optional stress mode: `engine.shuffle_symbols=true` with fixed seed to expose ordering sensitivity (off by default).

______________________________________________________________________

## 15. Glossary (P1)

- **Adjusted price (total‑return):** Price series reflecting both splits and cash dividends.
- **Ex‑date:** First date trading without dividend entitlement; shorts owe dividend here.
- **MOC:** Market‑on‑Close; executes in closing auction.
- **Participation cap:** Fractional cap of market volume an order can consume in a bar.

______________________________________________________________________

## 16. Phase‑2 Backlog (not in P1)

- GTC/OPG/OPD/stop‑limit/pegged orders; multi‑venue microstructure.
- Broker emulation; smart routing; impact models.
- PIT reference data enforcement and universe builders.
- Lot‑aware inventory and CA share mutations.
- Extended analytics/tear sheets; drawdown stats; turnover; exposure analytics.
- FX/multi‑currency support.

______________________________________________________________________

## 17. Appendix — JSON Schemas (selected)

### 17.1 `Bar` (OHLCV only)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["ts", "symbol", "open", "high", "low", "close", "volume"],
  "properties": {
    "ts": {"type": "string", "pattern": "^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$"},
    "symbol": {"type": "string"},
    "open": {"type": "number"},
    "high": {"type": "number"},
    "low": {"type": "number"},
    "close": {"type": "number"},
    "volume": {"type": "integer", "minimum": 0}
  }
}
```

### 17.2 `AdjustmentEvent` (optional metadata)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["ts", "symbol", "event_type", "px_factor", "vol_factor"],
  "properties": {
    "ts": {"type": "string", "pattern": "^\d{4}-\d{2}-\d{2}$"},
    "symbol": {"type": "string"},
    "event_type": {"type": "string", "enum": ["CashDiv", "Split", "StockDiv", "SpinOff"]},
    "px_factor": {"type": "number"},
    "vol_factor": {"type": "number"},
    "metadata": {"type": "object"}
  }
}
```

### 17.3 `fills.csv` columns

```text
fill_id,order_id,execution_ts,symbol,side,qty,price,slip_bps,fees,participation,partial_index
```

### 17.4 `orders.csv` columns

```text
order_id,strategy_ts,symbol,side,qty,type,limit,stop,tif,state,remaining
```

______________________________________________________________________

## 18. Appendix — Worked Examples (abridged)

> All examples assume daily bars, conservative fill rules, and the fixture dataset.

### 18.1 Limit Buy Touches

- **Setup:** Limit Buy 100 @ 100.00; bar has `low=99.50`, `close=100.40`.
- **Rule:** fill at `min(limit, close)` = 100.00.
- **fills.csv**

```text
F001,O123,2020-05-12,MSFT,BUY,100,100.00,0,0.50,0.01,0
```

### 18.2 Buy Stop Triggers

- **Setup:** Stop Buy 100 @ 102.00; bar `high=103.00`, `close=102.80`.
- **Rule:** fill at `max(stop, close)` = 102.80.
- **fills.csv**

```text
F002,O124,2020-06-03,AMZN,BUY,100,102.80,0,0.50,0.01,0
```

### 18.3 Market‑On‑Close (MOC)

- **Setup:** MOC Sell 50; close = 120.10; `moc.slip_bps=5`.
- **Rule:** sell price = 120.10 − 5bp = 120.04.
- **fills.csv**

```text
F003,O125,2021-03-29,AAPL,SELL,50,120.04,5,0.50,0.00,0
```

### 18.4 Malformed Bar → Close‑Only Policy

- **Setup:** Bar has `high < max(open,close)`; policy `warn_use_close_only`.
- **Rule:** **no limit/stop evaluation**; Market/MOC allowed.
- **fills.csv** (Market next‑open from previous signal)

```text
F004,O126,2022-11-14,MSFT,BUY,75,245.30,0,0.50,0.00,0
```

### 18.5 Participation Cap with Residual Queue

- **Setup:** Buy 10,000; cap=10% and bar volume=60,000 → max fill 6,000; `queue_bars=2`.
- **Rule:** 6,000 fill on bar t; 4,000 carried to t+1 (and filled if cap allows).
- **fills.csv**

```text
F005,O127,2020-09-10,AMZN,BUY,6000,3101.20,0,3.00,0.10,0
F006,O127,2020-09-11,AMZN,BUY,4000,3110.50,0,2.00,0.07,1
```

______________________________________________________________________

## 19. Distribution & Usage (Installable Package + CLI)

### 19.1 Installation

- **PyPI name:** `qtrader`

- **Requires:** Python 3.13+

- **Install:**

  - `pip install qtrader`
  - or `uv pip install qtrader`

- **SemVer:** Public API follows semantic versioning. Breaking changes bump **major** version.

### 19.2 Public Python API (stable in v1.x)

The end‑user **does not** modify `qtrader` internals. They import the public API, write strategies, and run backtests.

**Primary symbols:**

```python
from qtrader import Strategy, Context, Backtest, load_config, run_backtest
```

**Strategy contract (P1):**

```python
class Strategy:
    """
    Base strategy class. User strategies inherit from this.

    Strategy file is self-contained: includes both trading logic and config.

    Lifecycle hooks (in order):
    1. on_init()  - Called before warmup; register custom indicators
    2. [warmup phase - if enabled]
    3. on_start() - Called after warmup; strategy setup
    4. on_bar()   - Called for each bar during trading
    5. on_fill()  - Called after each fill
    6. on_end()   - Called at backtest end
    """
    # Optional: Strategy configuration (Pydantic model)
    config: Optional[BaseModel] = None

    def on_init(self, ctx: Context) -> None: ...      # optional; before warmup
    def on_start(self, ctx: Context) -> None: ...     # optional; after warmup
    def on_bar(self, bar: Bar, ctx: Context) -> None: ...  # required
    def on_fill(self, fill, ctx: Context) -> None: ...# optional
    def on_end(self, ctx: Context) -> None: ...       # optional
```

**Context API (extended for multi-dataset):**

```python
class Context:
    # Trading API
    def buy_market(self, symbol: str, qty: int) -> None: ...
    def sell_market(self, symbol: str, qty: int) -> None: ...
    def buy_limit(self, symbol: str, qty: int, limit: Decimal) -> None: ...
    def sell_limit(self, symbol: str, qty: int, limit: Decimal) -> None: ...
    def buy_stop(self, symbol: str, qty: int, stop: Decimal) -> None: ...
    def sell_stop(self, symbol: str, qty: int, stop: Decimal) -> None: ...
    def buy_moc(self, symbol: str, qty: int) -> None: ...
    def sell_moc(self, symbol: str, qty: int) -> None: ...

    # Position/Portfolio API
    def get_position(self, symbol: str) -> int: ...
    def get_cash(self) -> Decimal: ...
    def get_equity(self) -> Decimal: ...

    # Multi-dataset API (Phase 1A)
    def get_data(self, dataset: str, symbol: str, ts: datetime, field: str, default: Any = None) -> Any: ...
    def get_bars(self, dataset: str, symbol: str, start: datetime, end: datetime) -> List[Bar]: ...
    def has_data(self, dataset: str, symbol: str, ts: datetime) -> bool: ...
    def list_datasets(self) -> List[str]: ...
    def get_dataset_info(self, dataset: str) -> dict: ...

    # Indicator API (Phase 1 - Stage 6A)
    @property
    def ind(self) -> IndicatorManager: ...
    """Access indicator framework (ctx.ind.sma(), ctx.ind.rsi(), etc.)"""

    # Bar history API (for indicators)
    def current_bar(self, symbol: str) -> Bar | None: ...
    def get_bar_history(self, symbol: str, lookback: int) -> List[Bar]: ...

    # Indicator tracking & crossover helpers (Phase 1 - Stage 6A)
    def _track_indicator(self, symbol: str, key: str, value: Any) -> None: ...
    def crossed_above(self, symbol: str, key1: str, key2: str) -> bool: ...
    def crossed_below(self, symbol: str, key1: str, key2: str) -> bool: ...
    def crossed_above_threshold(self, symbol: str, key: str, threshold: float) -> bool: ...
    def crossed_below_threshold(self, symbol: str, key: str, threshold: float) -> bool: ...
```

**Self-contained strategy example:**

```python
# strategies/sentiment_sma.py
from qtrader import Strategy, Context
from qtrader.models.bar import Bar
from pydantic import BaseModel, Field
from decimal import Decimal

class SentimentSMAConfig(BaseModel):
    """Strategy configuration with defaults."""
    fast_period: int = Field(default=20, description="Fast SMA period")
    slow_period: int = Field(default=50, description="Slow SMA period")
    sentiment_threshold: float = Field(default=0.7, description="Min sentiment score")
    position_size: int = Field(default=100, description="Shares per trade")

class SentimentSMA(Strategy):
    """SMA crossover with sentiment filter."""

    # Default config (can be overridden via CLI --set)
    config = SentimentSMAConfig()

    def __init__(self, config: SentimentSMAConfig = None):
        self.config = config or self.config

    def on_bar(self, bar: Bar, ctx: Context):
        # Get sentiment from auxiliary dataset
        sentiment = ctx.get_data("news", bar.symbol, bar.ts, "sentiment_score", default=0.5)

        # Only trade if sentiment is positive
        if sentiment < self.config.sentiment_threshold:
            return

        # SMA crossover logic
        fast = ctx.ind.sma(bar.symbol, self.config.fast_period)
        slow = ctx.ind.sma(bar.symbol, self.config.slow_period)

        if ctx.just_crossed_above(fast, slow):
            ctx.buy_market(bar.symbol, self.config.position_size)
        elif ctx.just_crossed_below(fast, slow):
            ctx.sell_market(bar.symbol, self.config.position_size)
```

**Config precedence:** `CLI --set` overrides > YAML file > package defaults.

### 19.3 Command Line Interface (CLI)

**Main entrypoint:** `qtrader`

**Design principle:** Strategy files are **self-contained** (code + config). Data config YAML is **system configuration only** (data sources, adapters, validation).

- **Backtest:**

```bash
qtrader backtest \
  --strategy strategies/sentiment_sma.py \
  --data configs/multi_dataset.yaml \
  --out ./runs/exp1 \
  --set fast_period=10 --set sentiment_threshold=0.8 \
  --warmup
```

- `--strategy PATH`: Path to self-contained strategy Python file (required)
- `--data PATH`: Path to data configuration YAML (optional, uses defaults if omitted)
- `--out PATH`: Output directory for results (required)
- `--set KEY=VALUE`: Override strategy config parameters (optional, multiple allowed)
- `--warmup`: Enable indicator warmup (optional, overrides config)
- `--warmup-bars N`: Set explicit warmup period (optional, requires --warmup)

**Key design decisions:**

- Strategy file is self-contained: trading logic + default configuration
- Data config YAML contains only system settings: data sources, adapters, validation
- No need to specify `module:ClassName` - CLI auto-discovers Strategy class
- Single dataset or multi-dataset transparent to CLI (based on config structure)

**Examples:**

```bash
# Basic usage (uses default data config)
qtrader backtest --strategy strategies/buy_hold.py --out results/run1/

# With data config
qtrader backtest \
  --strategy strategies/sma_cross.py \
  --data configs/algoseek_daily.yaml \
  --out results/sma/

# With parameter overrides (for tuning)
qtrader backtest \
  --strategy strategies/sentiment_sma.py \
  --data configs/multi_dataset.yaml \
  --out results/sentiment_fast/ \
  --set fast_period=10 \
  --set slow_period=30 \
  --set sentiment_threshold=0.8

# Multi-dataset: price + sentiment + factors
qtrader backtest \
  --strategy strategies/factor_composite.py \
  --data configs/multi_dataset_factors.yaml \
  --out results/factors/
```

- Exit code **0** on success; non‑zero on validation/fill errors

- **Validate data (no trading):**

```bash
qtrader validate-data --data configs/algoseek_daily.yaml
```

- Runs dataset integrity checks (§3.3/§3.4) and reports policy actions
- Validates OHLC relationships, frequency, monotonic timestamps
- Reports malformed bars according to `ohlc_policy`

### 19.4 Strategy Discovery & Packaging

- Users may keep strategies in any repo. The CLI imports with `--strategy` using Python’s module path.
- For teams, publish internal strategy packages (e.g., `pip install bank_qi_strats`) and reference `--strategy bank_qi_strats.alpha:MyAlpha`.

### 19.5 API Stability & Support

- Only the **public symbols** documented above are stable in v1.x. Internal modules under `qtrader.engine.*` are not part of the public API.
- Deprecations are announced one minor version in advance.

______________________________________________________________________

## 20. Interactive Debugging & Development Workflow

### 20.1 Design Philosophy

**Core requirement:** Quants must be able to **step through strategies bar-by-bar** using standard Python debuggers (pdb, ipdb, VS Code, PyCharm) with full introspection of:

- Current bar values (OHLCV)
- All indicator values
- Portfolio state (positions, cash, NAV)
- Pending orders and fills
- Auxiliary dataset values

**No special debug mode required** — strategies are plain Python classes that work seamlessly with all Python debugging tools.

### 20.2 Standard Python Debugging

**VS Code / PyCharm:**

```python
# strategies/my_strategy.py
from qtrader import Strategy, Context
from qtrader.models.bar import Bar

class MyStrategy(Strategy):
    def on_bar(self, bar: Bar, ctx: Context):
        # Set breakpoint here (VS Code: click gutter, PyCharm: Ctrl+F8)
        fast_sma = ctx.ind.sma(bar.symbol, 20)
        slow_sma = ctx.ind.sma(bar.symbol, 50)

        # Inspect variables in debugger:
        # - bar.close, bar.volume
        # - fast_sma, slow_sma
        # - ctx.get_position(bar.symbol)
        # - ctx.get_cash()

        if fast_sma > slow_sma:
            ctx.buy_market(bar.symbol, 100)
```

**Run with debugger:**

```bash
# VS Code: F5 (Run and Debug)
# PyCharm: Shift+F9 (Debug)
# CLI with pdb:
python -m pdb -m qtrader.cli backtest --strategy strategies/my_strategy.py --out debug_run/
```

**Interactive pdb/ipdb:**

```python
# Insert breakpoint in strategy code
import ipdb; ipdb.set_trace()

# Or use Python 3.7+ built-in
breakpoint()
```

### 20.3 Context Debug API

**Additional introspection methods on `Context` for debugging:**

```python
class Context:
    # Existing trading API...

    # Debug introspection (read-only)
    def debug_state(self) -> dict:
        """
        Get complete engine state snapshot for current bar.

        Returns:
            {
                'bar': {'ts': '2023-01-15', 'symbol': 'AAPL', 'close': 150.50, ...},
                'portfolio': {
                    'cash': 50000.00,
                    'nav': 75000.00,
                    'equity': 25000.00,
                    'positions': {'AAPL': {'qty': 100, 'avg_price': 145.00, 'market_value': 15050.00}}
                },
                'orders': {
                    'pending': [{'id': 'O123', 'symbol': 'MSFT', 'side': 'BUY', 'qty': 50, ...}],
                    'filled': [{'id': 'O122', 'symbol': 'AAPL', 'side': 'BUY', 'qty': 100, ...}]
                },
                'indicators': {
                    'AAPL': {'sma_20': 148.50, 'sma_50': 145.00, 'rsi_14': 65.5}
                },
                'aux_data': {
                    'news': {'sentiment_score': 0.75, 'article_count': 12},
                    'factors': {'value_z': 1.5, 'momentum_z': 0.8}
                }
            }
        """
        ...

    def debug_indicators(self, symbol: str = None) -> dict:
        """Get all indicator values for symbol (or all symbols if None)."""
        ...

    def debug_orders(self, state: str = None) -> List[dict]:
        """Get orders filtered by state (PENDING/FILLED/EXPIRED/CANCELED/ALL)."""
        ...

    def debug_fills(self, limit: int = 10) -> List[dict]:
        """Get last N fills."""
        ...

    def debug_bar_history(self, symbol: str, lookback: int = 10) -> List[Bar]:
        """Get last N bars for symbol."""
        ...
```

**Usage in debugger:**

```python
# At breakpoint in on_bar():
(pdb) pp ctx.debug_state()
{
    'bar': {'ts': '2023-01-15', 'symbol': 'AAPL', 'close': Decimal('150.50'), ...},
    'portfolio': {'cash': Decimal('50000.00'), 'nav': Decimal('75000.00'), ...},
    ...
}

(pdb) pp ctx.debug_indicators('AAPL')
{'sma_20': 148.50, 'sma_50': 145.00, 'rsi_14': 65.5, 'bbands_upper': 155.00}

(pdb) pp ctx.debug_orders('PENDING')
[{'id': 'O123', 'symbol': 'MSFT', 'side': 'BUY', 'qty': 50, 'type': 'LIMIT', 'limit': 250.00}]
```

### 20.4 Interactive Backtesting (Python API)

**For Jupyter notebooks and interactive sessions:**

```python
from qtrader import Backtest
from strategies.my_strategy import MyStrategy

# Create backtest
bt = Backtest(
    strategy=MyStrategy(),
    data_config="configs/algoseek_daily.yaml",
    output_dir="debug_run/"
)

# Run bar-by-bar with manual control
bt.setup()  # Load data, initialize strategy

# Step through bars manually
bar = bt.next_bar()  # Returns Bar or None if done
print(f"Bar: {bar.symbol} @ {bar.ts} close={bar.close}")
print(f"Cash: {bt.ctx.get_cash()}")
print(f"Positions: {bt.ctx.debug_state()['portfolio']['positions']}")

# Continue stepping
while bar := bt.next_bar():
    print(f"{bar.ts} {bar.symbol}: {bar.close}")
    # Can inspect bt.ctx at any point

# Or run all at once
bt.run()  # Runs remaining bars
bt.finalize()  # Writes output files
```

### 20.5 Debug Logging

**Structured logging with debug levels:**

```yaml
# In config or CLI flag
logging:
  level: DEBUG              # DEBUG | INFO | WARNING | ERROR
  output: console           # console | file | both
  file: runs/debug.log
  format: pretty            # pretty | json
  modules:
    qtrader.engine: DEBUG   # Detailed execution logs
    qtrader.adapters: INFO  # Data loading logs
    strategy: DEBUG         # Strategy-specific logs
```

**Strategy logging:**

```python
class MyStrategy(Strategy):
    def on_bar(self, bar: Bar, ctx: Context):
        # Use standard Python logging
        self.logger.debug(f"Processing {bar.symbol} @ {bar.ts}")

        fast = ctx.ind.sma(bar.symbol, 20)
        slow = ctx.ind.sma(bar.symbol, 50)

        self.logger.debug(f"SMA(20)={fast:.2f}, SMA(50)={slow:.2f}")

        if fast > slow:
            self.logger.info(f"BUY signal: {bar.symbol}")
            ctx.buy_market(bar.symbol, 100)
```

**CLI debug mode:**

```bash
# Enable debug logging
qtrader backtest \
  --strategy strategies/my_strategy.py \
  --data configs/algoseek_daily.yaml \
  --out debug_run/ \
  --log-level DEBUG \
  --log-output both
```

### 20.6 Conditional Breakpoints

**Stop at specific date/symbol:**

```python
class MyStrategy(Strategy):
    def on_bar(self, bar: Bar, ctx: Context):
        # Breakpoint on specific condition
        if bar.symbol == "AAPL" and bar.ts.date() == date(2023, 1, 15):
            breakpoint()  # Stop here

        # Or use logging
        if bar.ts >= datetime(2023, 1, 15):
            self.logger.setLevel(logging.DEBUG)
```

**CLI date range filtering (for faster iteration):**

```bash
# Run only specific date range for debugging
qtrader backtest \
  --strategy strategies/my_strategy.py \
  --data configs/algoseek_daily.yaml \
  --out debug_run/ \
  --start-date 2023-01-10 \
  --end-date 2023-01-20 \
  --symbols AAPL,MSFT
```

### 20.7 Visualization During Development

**Plot indicators and signals (optional dependency: matplotlib):**

```python
from qtrader import Backtest
import matplotlib.pyplot as plt

bt = Backtest(strategy=MyStrategy(), data_config="config.yaml")
bt.run()

# Access internal state for plotting
df = bt.get_bar_dataframe("AAPL")  # All bars for symbol
ind = bt.get_indicator_dataframe("AAPL")  # All indicator values

plt.figure(figsize=(14, 7))
plt.plot(df['ts'], df['close'], label='Close')
plt.plot(ind['ts'], ind['sma_20'], label='SMA(20)')
plt.plot(ind['ts'], ind['sma_50'], label='SMA(50)')

# Mark fills
fills = bt.get_fills_dataframe(symbol="AAPL")
buys = fills[fills['side'] == 'BUY']
sells = fills[fills['side'] == 'SELL']
plt.scatter(buys['ts'], buys['price'], marker='^', color='g', s=100, label='Buy')
plt.scatter(sells['ts'], sells['price'], marker='v', color='r', s=100, label='Sell')

plt.legend()
plt.show()
```

### 20.8 Testing with Synthetic Bars

**Create minimal bar sequences for unit testing:**

```python
from qtrader.models.bar import Bar
from decimal import Decimal
from datetime import datetime, timedelta
import pytz

def test_my_strategy():
    """Test strategy with synthetic bars."""
    strategy = MyStrategy()
    ctx = MockContext()  # Test fixture

    # Create synthetic bars
    tz = pytz.timezone("America/New_York")
    bars = [
        Bar(
            ts=datetime(2023, 1, i, tzinfo=tz),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=1000000
        )
        for i in range(1, 11)  # 10 days
    ]

    # Run strategy on synthetic bars
    for bar in bars:
        strategy.on_bar(bar, ctx)

    # Assert strategy behavior
    assert len(ctx.orders) == 1
    assert ctx.orders[0]['side'] == 'BUY'
```

### 20.9 Debug Output Files

**Enhanced output when debug mode enabled:**

```yaml
# Standard outputs:
runs/exp1/
  performance.json
  positions_daily.csv
  orders.csv
  fills.csv
  run.json

# Additional debug outputs:
  debug/
    bars.csv                      # All processed bars
    indicators.csv                # All indicator values per symbol/bar
    portfolio_snapshots.csv       # Portfolio state every bar
    order_state_transitions.csv   # Order state changes with timestamps
    execution_log.jsonl           # Detailed execution events
    strategy_log.txt              # Strategy logger output
```

**Enable debug output:**

```bash
qtrader backtest \
  --strategy strategies/my_strategy.py \
  --out debug_run/ \
  --debug-output
```

### 20.10 Common Debugging Workflows

#### Scenario 1: "Why didn't my order fill?"

```python
# At breakpoint after expected fill
(pdb) pp ctx.debug_orders('PENDING')
[{'id': 'O123', 'symbol': 'MSFT', 'type': 'LIMIT', 'limit': 250.00, 'remaining': 100}]

(pdb) pp bar
Bar(ts=2023-01-15, symbol='MSFT', high=Decimal('248.00'), low=Decimal('245.00'), close=Decimal('247.00'))

# Ah! High (248.00) < limit (250.00), so limit buy didn't touch
```

#### Scenario 2: "Why is my indicator NaN?"

```python
# At breakpoint
(pdb) pp ctx.debug_indicators('AAPL')
{'sma_20': nan, 'sma_50': nan}

(pdb) pp ctx.debug_bar_history('AAPL', 25)
# Only 10 bars available! Need 20+ for SMA(20)
```

#### Scenario 3: "Check portfolio state at specific date"

```python
if bar.ts.date() == date(2023, 1, 15):
    state = ctx.debug_state()
    print(f"NAV: {state['portfolio']['nav']}")
    print(f"Positions: {state['portfolio']['positions']}")
    print(f"Cash: {state['portfolio']['cash']}")
    breakpoint()
```

### 20.11 Performance Profiling

**Profile strategy performance:**

```bash
# Use Python profiler
python -m cProfile -o profile.stats -m qtrader.cli backtest \
  --strategy strategies/my_strategy.py \
  --out profile_run/

# Analyze profile
python -m pstats profile.stats
```

**Memory profiling:**

```bash
# Install memory_profiler
pip install memory_profiler

# Profile strategy
python -m memory_profiler -m qtrader.cli backtest \
  --strategy strategies/my_strategy.py \
  --out mem_profile_run/
```

______________________________________________________________________
