# Equities Backtesting Engine — Specification (Phase 1, **v1.0**)

**Owner:** Javier
**Readers:** Quant researchers, software engineers, data engineers
**Version:** 1.0 — Approved baseline for implementation
**Scope:** U.S. equities, bar‑based backtests (1m–1d). Deterministic by design.

---

## 1. Purpose & Non‑Goals

### 1.1 Purpose

Build a **deterministic**, **auditable**, and **extensible** equities backtesting engine with realistic (but configurable) execution modeling suitable for professional quants.

### 1.2 Non‑Goals (Phase 1)

* No live trading adapters; no broker emulation beyond simple fills.
* No PIT reference data enforcement (user responsible for PIT universes; engine logs assumptions).
* No FX/multi‑currency. USD only.
* No advanced risk/tearsheet metrics (Sharpe/Sortino/etc.) — P2.

---

## 2. Architecture Overview

### 2.1 Ports & Adapters

* **DataPort** → adapters (Algoseek Parquet, CSV, future vendors) normalize to canonical **Bar**.
* **ExecPort** → execution policies (fill rules, participation, slippage).
* **CostPort** → commissions/fees (per‑share + ticket min).
* **RiskPort** (minimal P1) → equity/notional checks.

### 2.2 Determinism

* Single‑threaded event loop per backtest run.
* Lexicographic tie‑break on equal timestamps across symbols.
* Fixed RNG seed for any stochastic components (not used in P1 by default).

### 2.3 Vendor Integration Model & Bar Contract (Authoritative)

**Design goal:** Accept **any vendor schema** → transform to a **canonical Bar** used end‑to‑end across the trade lifecycle (signals → orders → fills → accounting).

**Contract:**

```python
class Bar(NamedTuple):
    ts: datetime         # timezone-aware per data.timezone
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adj_reason: Optional[str] = None
    px_factor: Optional[Decimal] = None
    vol_factor: Optional[Decimal] = None
```

**Adapter interface:**

```python
class DataAdapter(Protocol):
    def can_read(self, source: URI) -> bool: ...
    def schema_version(self) -> str: ...
    def read_iter(self, source: URI, config: DataConfig) -> Iterable[RawRecord]: ...
    def normalize(self, raw: RawRecord, mapping: ColumnMap, config: DataConfig) -> Bar: ...
```

**Pipeline:** `Read RawRecord → Normalize (column_map, dtype, tz) → Validate (schema, frequency) → Emit Bar`.

**Schema registry:**

* Maintain `schemas/<vendor>/<dataset>/<version>.json` for RawRecord validation.
* `column_map` in config binds vendor fields → engine fields without code changes.

**Extensibility rules:**

* New vendor = new `DataAdapter` + schema file + unit tests; engine code unchanged.
* The **Bar** type is the **only** price/volume contract consumed by Exec/Cost/Risk.
* Optionally persist a **Standard Bar Extract** to disk (parquet) to speed reruns.

**Validation:**

* Strict type/required checks at adapter boundary.
* Frequency/monotonic checks after normalization.

**Versioning:**

* Adapters advertise `schema_version()`; runs persist this in `run.json` for reproducibility.

````
---

## 3. Dataset Alignment (Algoseek **Standard Adjusted** OHLC)

**Assumption:** Vendor bars are **total‑return adjusted** — both **dividends and splits** embedded via cumulative factors. Implications:

- **Long dividends:** **Do not** post separate long cash dividends; already in price path.  
- **Short dividends:** **Do** post **negative cash** on **ex‑date** for symbols held short when the adjustment reason indicates a cash dividend.
- **Splits / scrip / rights:** Already reflected in adjusted prices. P1 does not mutate share counts; lot/share CA mechanics are P2.

### 3.1 Canonical `Bar`
```json
{
  "ts": "YYYY‑MM‑DD[THH:MM]",
  "symbol": "AAPL",
  "open": 197.1200,
  "high": 198.5000,
  "low": 195.8000,
  "close": 197.9100,
  "volume": 32456789,
  "adj_reason": "DIV CashDiv|SD Subdiv|…",          // nullable
  "px_factor": 1.2345,                               // cumulative, optional
  "vol_factor": 0.8123                               // cumulative, optional
}
````

**Precision:** Prices serialized at `data.decimals.price` (default 4). Indicator math uses float64; ledger uses Decimal.

### 3.2 Data Config

```yaml
data:
  mode: standard_adjusted          # total‑return adjusted prices
  frequency: 1d                    # 1m|5m|15m|1h|1d
  timezone: America/New_York
  column_map: {ts: trade_date, symbol: ticker}
  strict_frequency: true           # raise if mismatch
  decimals: {price: 4, cash: 4}
  source_tag: "algoseek-standard-adjusted"
```

### 3.3 Integrity Checks

* Validate monotonic timestamps per symbol.
* Median delta must match `frequency` when `strict_frequency=true`.
* If `mode=standard_adjusted`, **disable long dividends** and **enable short dividend debits**.

### 3.4 Data Validation Policies (dataset‑specific)

Raw vendor bars can be wrong (e.g., `high < max(open,close)` or `low > min(open,close)`). Validation and fallback are **dataset‑specific** and controlled via YAML.

**Validation rules:**

* `high ≥ max(open, close) − epsilon`
* `low ≤ min(open, close) + epsilon`
* `low ≤ high + epsilon`
* `volume ≥ 0`

`epsilon` is a small tolerance (default `0.0` for daily; vendors sometimes require `1e‑6` on intraday).

**Policy (per‑dataset):**

```yaml
data:
  validation:
    epsilon: 0.0
    ohlc_policy: strict_raise   # strict_raise | warn_skip_bar | warn_use_close_only
    close_only_fields: [close]  # when warn_use_close_only, only these fields are trusted
```

* **`strict_raise`**: raise an error and stop run on first malformed bar.
* **`warn_skip_bar`**: log a warning, **skip the bar** (no fills, orders remain pending).
* **`warn_use_close_only`** (aka *close‑only/brown‑bar mode*): log a warning, **trust only `close` (and fields in `close_only_fields`)** and treat the bar as **ineligible for intrabar touch logic** (see §4.1/§4.3 interaction). Execution that requires `high/low` is **disabled for that bar**; queued orders remain pending to next bar. Market/MOC still function.

**Run metadata:**

* Engine records counts per policy: `{malformed_bars: N, skipped: X, close_only: Y}` in `run.json` and emits first/last 10 offending `(date,symbol)` pairs.

---

### 3.5 Reference Dataset (Fixtures)

**Purpose:** Single source of truth for tests, examples, and golden files.

* **Location:** `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample` (project‑relative `data/` folder).
* **Format:** **Parquet**.
* **Partitioning:** by **`SecId`**. Path template: `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/SecId=<SecId>/*.parquet`.
* **Universe:** `MSFT`, `AMZN`, `AAPL`.
* **Date range:** **2019‑01‑01** to **2023‑12‑31** (inclusive).
* **Frequency:** `1d` (daily) for P1 fixtures.
* **Column map (example):** `{ ts: trade_date, symbol: ticker, open: open, high: high, low: low, close: close, volume: market_hours_volume }`.
* **Usage rules:**

  * All **unit/integration tests**, **reference strategies** (e.g., Buy‑and‑Hold, SMA Cross), and **golden results** MUST run against this fixture by default.
  * CI loads fixture the same way as users: through the **DataAdapter → Bar** normalization pipeline.
  * Any change to fixture or adapter **invalidates goldens**; bump golden version and store alongside `run.json`.

---

## 4. Orders & Execution

### 4.1 Order Types

* **Market** — Default for close‑based signals is **Next‑Bar‑Open** (no same‑bar close fills). Intrabar market fills allowed only when strategy is intrabar and the bar passes OHLC validation.
* **Market‑On‑Close (MOC)** — Executes on current bar’s close/auction with configurable auction slippage (bps). Allowed in all policies.
* **Limit** — Touch rules configurable; conservative by default. **Disabled for the current bar** if `ohlc_policy=warn_use_close_only` is applied because high/low cannot be trusted; order remains pending.
* **Stop (Stop→Market)** — Trigger then market with conservative rule by default. **Disabled for the current bar** if `ohlc_policy=warn_use_close_only` is applied; order remains pending.

### 4.2 Time‑In‑Force (TIF)

* Market/MOC ignore TIF (IOC by nature).
* Limit/Stop default **DAY** (expire at end of bar/day). GTC is P2.

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

* **Limit Buy:** if `low ≤ limit` then fill at `min(limit, close)`; else no fill.
* **Limit Sell:** if `high ≥ limit` then fill at `max(limit, close)`; else no fill.
* **Stop Buy:** if `high ≥ stop` then fill at `max(stop, close)` ± slippage.
* **Stop Sell:** if `low ≤ stop` then fill at `min(stop, close)` ± slippage.
* **MOC:** fill at close price ± `moc.slip_bps` (bp add for buys, subtract for sells).

**Governance:** The team **pins conservative** as default. Switching `limit_mode`/`stop_mode` to `optimistic` requires code review sign‑off and a change log entry in the repository.

### 4.4 Volume Participation & Partial Fills (ENFORCED) & Partial Fills (ENFORCED)

* Max shares fill per bar per order side: `cap = max_participation × bar.volume`.
* If requested qty > `cap`, engine **partially fills** up to `cap` and **queues residual** forward for up to `queue_bars` bars; residual expires afterward.
* Each partial is a distinct fill slice with its own costs/slippage.

### 4.5 Order Lifecycle & States

`SUBMITTED → (TRIGGERED for stops) → PARTIALLY_FILLED* → FILLED | EXPIRED | CANCELED`

---

## 5. Shorting, Borrow, and Dividends

```yaml
trading:
  allow_short: false              # default false
  borrow_rate_annual: 0.03        # flat annual; accrues EOD on short MV
```

* If `allow_short=false`, orders that would create net short are rejected.
* If `allow_short=true`:

  * **Borrow cost accrual:** each EOD: `cash -= |short_market_value| × (borrow_rate_annual/252)`.
  * **Short dividends:** on **ex‑date** when `adj_reason` indicates cash dividend and net short at bar close (EOD), post `cash -= |shares| × dividend_per_share`.
  * **Long dividends:** none (embedded in adjusted prices).

**Dividend source:** For P1, `dividend_per_share` is derived from vendor CA metadata or implied from price factors when available; if missing, engine logs a warning and skips cash posting (prices still reflect the event).

---

## 6. Calendar & Session Semantics

* No external session calendar in P1. “**Next‑open**” = next available bar in the dataset.
* Holidays/halts/early closes are implicit in the bar series; pending DAY orders roll to next bar.
* MOC executes on the current bar close (including early close bars).

---

## 7. Event Loop (Canonical)

``` python
for bar in dataset:
  emit(bar)
  strategy.on_bar(bar)                   # submit orders (strategy_ts = bar.ts)
  exec.evaluate_intrabar(bar)            # limit/stop touches; apply participation; partials
  exec.end_of_bar(bar)                   # MOC on current; schedule Market for next open
  ledger.apply_fills_and_costs()
  ledger.eod_accruals()                  # borrow, short dividends if ex‑date
  outputs.snapshot_if_eod()
  logs.write()
```

**Timestamps:** each fill slice records `strategy_ts` (submission) and `execution_ts` (fill). Engine clock = bar timestamps; no sub‑bar clock in P1.

---

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
```

---

## 9. Costs & Commissions

* **Per‑share:** `costs.per_share` × filled shares.
* **Ticket minimum:** apply once per order per bar slice (`fills.csv` shows enforced minimum flag).
* Costs are deducted from cash at the time of each fill.

### 9.1 Ledger Computation Order (authoritative)

For every fill slice: **(1) start from execution price**, **(2) apply slippage (bps) to price**, **(3) compute gross cash impact**, **(4) apply fees/commissions**, **(5) update positions and cash**, **(6) quantize ledger values to Decimal with configured `data.decimals`**.

---

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

* config snapshot, code version/hash, RNG seed, `data.source_tag`, universe text, warnings.

### 10.6 `signals.jsonl` (optional)

One JSON object per line containing strategy debug data.

---

## 11. Error Handling & Validation

* **Data frequency:** raise when `strict_frequency=true` and median delta mismatch.
* **NaN/negative prices or volumes:** raise with symbol/date detail.
* **Malformed OHLC bars:** apply `data.validation.ohlc_policy`:

  * `strict_raise` → error out with first offending `(date,symbol)`.
  * `warn_skip_bar` → warn and skip; count in `run.json`.
  * `warn_use_close_only` → warn and mark bar *close‑only*; disallow limit/stop evaluation on that bar and record in `run.json`.
* **High participation guardrail:** if `fills.max_participation > 0.20` and `fills.allow_high_participation=false`, **warn and clamp to 0.20** for the run; record in `run.json`.
* **Order rejections:** invalid qty/side, shorting disabled, TIF violations, etc.
* **Adjustment sanity:** warn if `adj_reason` missing on large discontinuities; configurable `adjustments.enforce: warn|raise`.

---

## 12. Testing Strategy (P1)

Testing must focus on functionality, we don,'t care about coverage. Keep it simple. 

### 12.1 Fixture Definition (authoritative)

* Dataset: **Parquet**, partitioned by **`SecId`**, under `./data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample`.
* Universe: **MSFT, AMZN, AAPL**.
* Date range: **2019‑01‑01 → 2023‑12‑31**.
* Frequency: **1d**.

### 12.2 Unit Tests

* Limit/Stop conservative vs optimistic fills (all four cases) using fixture bars.
* Market (next‑open) vs MOC with `moc.slip_bps` on fixture dates with high/low bounds.
* Participation cap with residual queue over multiple bars within fixture.
* Short dividend debit on known ex‑dates present in adjusted series; borrow accrual daily.
* Commission model with ticket minimum.
* Strict frequency and bad data guards using synthetic corrupted copies of fixture.
* **Malformed bar policies:**

  * Inject bars with `high < max(open,close)` and verify `strict_raise` aborts.
  * Inject bars with `low > min(open,close)` and verify `warn_skip_bar` keeps orders pending.
  * Inject either case and verify `warn_use_close_only` allows **Market/MOC** but **prevents limit/stop** fills for the offending bar.

### 12.3 Golden Files (Determinism)

* **Buy‑and‑Hold** each of MSFT, AMZN, AAPL → canonical NAV paths (no costs and with costs).
* **SMA Cross** on MSFT → canonical fills & PnL.
* Goldens are stored under `tests/goldens/<strategy>/<version>/` with `run.json` including adapter `schema_version()` and fixture hash.

### 12.4 CI Fixture Integrity

* Compute and store a **checksum/hash** of the fixture directories (e.g., per‑partition hash + global Merkle root) in repo (`tests/fixtures/manifest.json`) and verify at CI start.
* If mismatch: fail fast with guidance; include expected vs observed hash in logs.
* Persist the used fixture hash into `run.json` for auditability.

---

## 13. Performance & Numerics

* Indicator math in float64 for speed; ledger and serialized prices in Decimal (quantized to `data.decimals`).
* Target: 10 years × 3k symbols × 1d bars under 60s on a modern CPU (non‑parallel). Provide `numerics.mode: {float_indicators|full_decimal}` toggle (default `float_indicators`).

---

## 14. Developer Notes

### 14.1 Order of Operations (Bar)

1. Strategy computes and submits orders.
2. Intrabar evaluation for limit/stop (with participation).
3. End‑bar evaluation: MOC fill; Market scheduled for next bar open.
4. Ledger updates: positions, cash, PnL, costs.
5. EOD accruals: borrow; short dividends on ex‑date.
6. Output snapshots and logs.

### 14.2 Symbol Ordering & Reproducibility

* Per‑bar multi‑symbol processing uses **lexicographic** order for determinism.
* Optional stress mode: `engine.shuffle_symbols=true` with fixed seed to expose ordering sensitivity (off by default).

---

## 15. Glossary (P1)

* **Adjusted price (total‑return):** Price series reflecting both splits and cash dividends.
* **Ex‑date:** First date trading without dividend entitlement; shorts owe dividend here.
* **MOC:** Market‑on‑Close; executes in closing auction.
* **Participation cap:** Fractional cap of market volume an order can consume in a bar.

---

## 16. Phase‑2 Backlog (not in P1)

* GTC/OPG/OPD/stop‑limit/pegged orders; multi‑venue microstructure.
* Broker emulation; smart routing; impact models.
* PIT reference data enforcement and universe builders.
* Lot‑aware inventory and CA share mutations.
* Extended analytics/tear sheets; drawdown stats; turnover; exposure analytics.
* FX/multi‑currency support.

---

## 17. Appendix — JSON Schemas (selected)

### 17.1 `Bar` (simplified)

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
    "volume": {"type": "integer", "minimum": 0},
    "adj_reason": {"type": ["string", "null"]},
    "px_factor": {"type": ["number", "null"]},
    "vol_factor": {"type": ["number", "null"]}
  }
}
```

### 17.2 `fills.csv` columns

``` text
fill_id,order_id,execution_ts,symbol,side,qty,price,slip_bps,fees,participation,partial_index
```

### 17.3 `orders.csv` columns

``` text
order_id,strategy_ts,symbol,side,qty,type,limit,stop,tif,state,remaining
```

---

## 18. Appendix — Worked Examples (abridged)

> All examples assume daily bars, conservative fill rules, and the fixture dataset.

### 18.1 Limit Buy Touches

* **Setup:** Limit Buy 100 @ 100.00; bar has `low=99.50`, `close=100.40`.
* **Rule:** fill at `min(limit, close)` = 100.00.
* **fills.csv**

``` text
F001,O123,2020-05-12,MSFT,BUY,100,100.00,0,0.50,0.01,0
```

### 18.2 Buy Stop Triggers

* **Setup:** Stop Buy 100 @ 102.00; bar `high=103.00`, `close=102.80`.
* **Rule:** fill at `max(stop, close)` = 102.80.
* **fills.csv**

``` text
F002,O124,2020-06-03,AMZN,BUY,100,102.80,0,0.50,0.01,0
```

### 18.3 Market‑On‑Close (MOC)

* **Setup:** MOC Sell 50; close = 120.10; `moc.slip_bps=5`.
* **Rule:** sell price = 120.10 − 5bp = 120.04.
* **fills.csv**

``` text
F003,O125,2021-03-29,AAPL,SELL,50,120.04,5,0.50,0.00,0
```

### 18.4 Malformed Bar → Close‑Only Policy

* **Setup:** Bar has `high < max(open,close)`; policy `warn_use_close_only`.
* **Rule:** **no limit/stop evaluation**; Market/MOC allowed.
* **fills.csv** (Market next‑open from previous signal)

``` text
F004,O126,2022-11-14,MSFT,BUY,75,245.30,0,0.50,0.00,0
```

### 18.5 Participation Cap with Residual Queue

* **Setup:** Buy 10,000; cap=10% and bar volume=60,000 → max fill 6,000; `queue_bars=2`.
* **Rule:** 6,000 fill on bar t; 4,000 carried to t+1 (and filled if cap allows).
* **fills.csv**

``` text
F005,O127,2020-09-10,AMZN,BUY,6000,3101.20,0,3.00,0.10,0
F006,O127,2020-09-11,AMZN,BUY,4000,3110.50,0,2.00,0.07,1
```
