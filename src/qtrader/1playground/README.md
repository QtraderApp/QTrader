# QTrader Playground - Data Layer Testing

This directory contains tools for generating golden output and validating the data layer implementation.

## Overview

The data layer transformation pipeline:

```
Raw Algoseek Data → AlgoseekBar → AlgoseekPriceSeries → CanonicalPriceSeries (3 modes)
```

**Three adjustment modes:**

1. **Unadjusted**: Raw prices as traded, actual volume
1. **Adjusted**: Split-adjusted prices (backward adjustment)
1. **Total Return**: Forward compounding with dividend reinvestment

## Files

### Core Scripts

#### `generate_golden_output.py`

Generates golden output JSON file from Algoseek data.

**Usage:**

```bash
python src/qtrader/1playground/generate_golden_output.py
```

**Default configuration:**

- Symbol: AAPL
- Date range: 2020-08-01 to 2020-09-01
- Includes: Dividend ($1.00 on 8/7) and 4:1 split (8/31)
- Output: `golden_output_new.json`

**Customization:** Edit the configuration variables at the top of the script:

```python
DATA_PATH = Path("...")  # Path to parquet data
START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
SYMBOL = "AAPL"
OUTPUT_FILE = Path("...")  # Output path
```

**Output structure:**

```json
{
  "metadata": {
    "symbol": "AAPL",
    "start_date": "2020-08-01",
    "end_date": "2020-09-01",
    "bar_count": 22,
    "corporate_events": {
      "dividends": [...],
      "splits": [...]
    }
  },
  "series": {
    "unadjusted": { "bars": [...] },
    "adjusted": { "bars": [...] },
    "total_return": { "bars": [...] }
  }
}
```

#### `test_validate_golden.py`

Validation framework for comparing test output against golden output.

**Key functions:**

- `validate_against_golden(test_data, golden_path)` - Full validation
- `compare_bars(golden_bars, test_bars, mode, result)` - Bar-by-bar comparison
- `validate_key_points(test_data, golden_path)` - Quick sanity check

**Validation tolerances:**

- Prices: ±$0.01
- Volume: Exact match
- Dividends: ±$0.0001

**Usage in tests:**

```python
from test_validate_golden import validate_against_golden

# Generate test data
test_data = {
    "metadata": {...},
    "series": {
        "unadjusted": {"bars": [...]},
        "adjusted": {"bars": [...]},
        "total_return": {"bars": [...]}
    }
}

# Validate
passed = validate_against_golden(test_data, Path("golden_output_new.json"))
```

#### `test_validation_script.py`

Test script to verify the validation framework works correctly.

**Usage:**

```bash
python src/qtrader/1playground/test_validation_script.py
```

Generates fresh data and validates it against golden output - should show 100% match.

### Legacy/Archive Files

- `test_golden_output.py` - Original golden output generator (superseded)
- `test_new_data_layer.py` - Original validation test
- `golden_output.json` - Original golden output (kept for reference)

## Workflow

### 1. Generate Golden Output (One-time)

```bash
python src/qtrader/1playground/generate_golden_output.py
```

This creates `golden_output_new.json` capturing the current correct implementation.

### 2. Make Changes to Data Layer

Edit files in:

- `src/qtrader/models/canonical_bar.py`
- `src/qtrader/models/vendors/algoseek/`

### 3. Validate Changes

```bash
# Run validation test
python src/qtrader/1playground/test_validation_script.py

# Or use in your own test:
from test_validate_golden import validate_against_golden

test_data = generate_test_data()  # Your implementation
passed = validate_against_golden(test_data, golden_path)
```

### 4. Review Results

The validation reports:

- ✅ All tests passed - Changes are safe
- ❌ Tests failed - Shows specific mismatches

## Testing Data

**AAPL 2020-08-01 to 2020-09-01:**

- 22 trading days
- 1 dividend: $1.00 on 2020-08-07
- 1 split: 4:1 on 2020-08-31

**Why this data?**

- Real historical event (Apple's 2020 split)
- Includes both dividend and split
- Short enough to review manually
- Complex enough to catch bugs

## Corporate Event Detection

The data layer automatically detects:

**Dividends:**

- Cash dividends (CashDiv)
- Script dividends (ScriptDiv)
- Script dividend differentials (ScriptDivDiff)

**Splits:**

- Forward splits (e.g., 4:1)
- Reverse splits (e.g., 1:5)
- Bonus issues (BonusSame)
- Consolidations (Cons)

## Adjustment Formulas

### Unadjusted

```
Price: Raw as-traded
Volume: Actual traded volume
Dividend: Recorded on ex-date
```

### Adjusted (Split-adjusted)

```
Price = Unadjusted / (LastVolFactor / CurrentVolFactor)
Volume = Unadjusted * (LastVolFactor / CurrentVolFactor)
Dividend = Adjusted for splits
```

### Total Return

```
TR_t = TR_{t-1} × (UnAdj_t × Split_t + Div_t) / UnAdj_{t-1}
Volume = Unadjusted / CumulativeSplitRatio
Dividend = None (embedded in prices)
```

## Validation Metrics

For AAPL 2020-08-01 to 2020-09-01:

| Mode         | First Close | Last Close | Return    |
| ------------ | ----------- | ---------- | --------- |
| Unadjusted   | $435.75     | $134.18    | -69.21%\* |
| Adjusted     | $108.94     | $134.18    | +23.17%   |
| Total Return | $435.75     | $537.93    | +23.45%   |

\* Negative return is due to 4:1 split - not an actual loss

**Volume on split date (8/31):**

- Unadjusted: 210M (actual traded)
- Adjusted: 210M (post-split units)
- Total Return: 52M (pre-split units, 210M / 4)

## Troubleshooting

**Q: Validation fails with price mismatches** A: Check if corporate events are detected correctly. Use `get_split_ratio()` and `get_dividend_amount()` methods.

**Q: Volume doesn't match** A: Verify split ratio calculation. For forward splits, volume should multiply (adjusted) or divide (total return).

**Q: Golden output file not found** A: Run `generate_golden_output.py` first to create it.

**Q: Import errors with test_validate_golden** A: The folder name starts with "1" which complicates imports. Use `sys.path.insert()` as shown in `test_validation_script.py`.

## Maintenance

**When to regenerate golden output:**

- Bug fix changes numerical results
- New corporate event type support
- Formula corrections

**When to update validation:**

- New adjustment mode added
- Additional validation checks needed
- Tolerance levels need adjustment

## See Also

- `docs/DATA_LAYER_TEST_COVERAGE.md` - Comprehensive test documentation
- `docs/price_series_architecture_refactoring.md` - Architecture design
- `tests/unit/models/test_canonical_bar.py` - Unit tests
- `tests/unit/models/test_vendors_algoseek.py` - Vendor tests
- `tests/integration/test_data_layer_corporate_events.py` - Integration tests
