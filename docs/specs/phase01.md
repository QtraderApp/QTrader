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

## 13. Performance & Numerics

**Precision:**

- Indicator math: float64 for speed
- Ledger & serialized prices: Decimal (quantized to `data.decimals`)
- Configuration toggle: `numerics.mode: {float_indicators|full_decimal}` (default `float_indicators`)

**Performance Targets:**

- 10 years × 3k symbols × 1d bars under 60s on modern CPU (non-parallel)
- Incremental indicator updates: O(1) per bar for rolling windows (SMA, EMA)

**Indicators Framework (Stage 6A - Implemented):**

- Built-in indicators: SMA, EMA, Bollinger Bands, ATR, RSI, MACD
- Custom indicators: User-defined via `Indicator[T]` base class
- Helper functions: 13 utilities for crossover/threshold/divergence detection
- Warmup support: Automatic pre-computation before trading starts
- Access API: `ctx.ind.sma(symbol, period=20)`

*See `docs/indicators_architecture.md` and `examples/sma_crossover_strategy.py` for implementation details.*

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

## 15. Risk Management (Phase 1)

**Architecture:** Centralized gatekeeper between strategy signals and order execution.

**Signal Model (Stage 5B - Implemented):**

```python
class Signal(NamedTuple):
    """Trading intent from strategy (pre-sizing)."""
    signal_id: str
    strategy_ts: datetime
    symbol: str
    signal_type: SignalType          # ENTRY_LONG | ENTRY_SHORT | EXIT_LONG | EXIT_SHORT
    direction: SignalDirection        # LONG | SHORT | FLAT

    # Sizing hints (optional)
    target_qty: Optional[int] = None
    target_weight: Optional[Decimal] = None  # Portfolio weight (0.0-1.0)

    # Order preferences
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    tif: TimeInForce = TimeInForce.DAY

    # Risk context
    conviction: Decimal = Decimal("1.0")  # Confidence (0.0-1.0)
```

**Risk Policy Configuration:**

```yaml
risk:
  # Position sizing
  sizing_method: portfolio_percent  # fixed_quantity | fixed_value | portfolio_percent | risk_percent
  default_position_size: 0.05       # 5% of equity per position

  # Concentration limits
  max_position_pct: 0.20            # Max 20% in single position
  max_positions: 10                 # Max concurrent positions

  # Leverage & exposure
  max_gross_exposure: 1.0           # Max 100% gross (long + |short|)
  max_net_exposure: 1.0             # Max 100% net (long - |short|)
  allow_shorting: false

  # Safety
  cash_reserve_pct: 0.05            # Keep 5% cash reserve
  reject_on_insufficient_cash: true
  reject_on_concentration_breach: true
```

**Evaluation Flow:**

1. Validate signal direction (shorting allowed?)
1. Check portfolio constraints (leverage, exposure)
1. Calculate position size using `sizing_method`
1. Apply concentration limits
1. Check cash availability
1. Return `RiskDecision` (approved/rejected + sized qty)

**Strategy Integration:**

```python
# Strategy emits signals (intent)
signal = Signal(
    signal_id="sig-1",
    strategy_ts=bar.ts,
    symbol="AAPL",
    signal_type=SignalType.ENTRY_LONG,
    direction=SignalDirection.LONG,
    target_weight=Decimal("0.10")  # Want 10% of equity
)

# Context routes to risk manager for sizing
order_id = ctx.submit_signal(signal)  # Returns order_id if approved, None if rejected
```

**Output Artifacts:**

- `signals.jsonl`: All signals with approval status, rejections reasons, sized quantities
- `risk_summary.json`: Approval rates, rejection breakdown, sizing statistics

*See `docs/risk_management_guide.md` and `examples/risk_signal_example.py` for implementation details.*

______________________________________________________________________

## 16. Glossary (P1)

- **Adjusted price (total‑return):** Price series reflecting both splits and cash dividends.
- **Ex‑date:** First date trading without dividend entitlement; shorts owe dividend here.
- **MOC:** Market‑on‑Close; executes in closing auction.
- **Participation cap:** Fractional cap of market volume an order can consume in a bar.
- **Signal:** Trading intent from strategy before position sizing (Phase 1, Stage 5B+).
- **Risk decision:** RiskManager evaluation result (approved/rejected + sized quantity).

______________________________________________________________________

## 17. Phase‑2 Backlog (not in P1)

- GTC/OPG/OPD/stop‑limit/pegged orders; multi‑venue microstructure.
- Broker emulation; smart routing; impact models.
- PIT reference data enforcement and universe builders.
- Lot‑aware inventory and CA share mutations.
- Extended analytics/tear sheets; drawdown stats; turnover; exposure analytics.
- FX/multi‑currency support.
- Advanced risk sizing (volatility target, Kelly criterion, risk parity).
- Multi-strategy orchestration with signal prioritization.
- Dynamic risk adjustment based on market regime.

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
