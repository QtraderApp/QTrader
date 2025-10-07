# Golden Test Approach - README

## Overview

This directory uses a **golden test** approach for refactoring the price series architecture. We capture the current implementation's output as the "golden standard" and validate that the refactored code produces identical results.

## 🎯 Workflow

### Phase 1: ✅ Generate Golden Output (COMPLETED)

```bash
python src/qtrader/1playground/test_golden_output.py
```

**Output**: `golden_output.json` containing:

- 24 bars of AAPL data (2020-08-01 to 2020-09-03)
- 3 series: unadjusted, adjusted, total_return
- Corporate events: dividend on 8/7, 4:1 split on 8/31

### Phase 2: 📝 Refactor Architecture (TODO)

Refactor `debug_algoseek_bars.py` into service layers:

1. **Data Models** - Pure Pydantic classes
1. **Corporate Events Extractor** - Parse vendor events
1. **Price Adjuster** - Split adjustment math
1. **Total Return Calculator** - Investment simulation

See `../../docs/price_series_architecture_refactoring.md` for details.

### Phase 3: ✅ Validate Refactored Code (TODO)

```python
from test_validate_golden import validate_against_golden

# Generate output from refactored implementation
test_data = your_refactored_implementation()

# Validate against golden
passed = validate_against_golden(test_data, Path("golden_output.json"))
```

## 📁 Files

| File                      | Purpose                         |
| ------------------------- | ------------------------------- |
| `debug_algoseek_bars.py`  | Current working implementation  |
| `test_golden_output.py`   | Generates golden output         |
| `golden_output.json`      | Golden output (source of truth) |
| `test_validate_golden.py` | Validation test framework       |
| `ARCHITECTURE.md`         | Original architecture doc       |
| `GOLDEN_TEST.md`          | This file                       |

## 🔍 What Gets Validated

### Exact Matches Required

- Bar count (must be exactly 24)
- Dates (exact string match)
- Volume (exact integer match)

### Tolerance Allowed

- OHLC prices: ±$0.01 (floating point rounding)
- Dividends: ±$0.0001 (decimal precision)

### Key Test Cases

**2020-08-07 (Dividend Day)**

```
Unadjusted:  Close=$444.45, Vol=46,760,815, Div=$0.7999
Adjusted:    Close=$111.11, Vol=187,043,259, Div=$0.20
TotalReturn: Close=$445.25, Vol=46,760,815
```

**2020-08-28 (Pre-Split)**

```
Unadjusted:  Close=$499.23, Vol=44,260,263
Adjusted:    Close=$124.81, Vol=177,041,051  (÷4 for price, ×4 for volume)
TotalReturn: Close=$500.13, Vol=44,260,263   (forward compounded)
```

**2020-08-31 (Split Day)**

```
Unadjusted:  Close=$129.04, Vol=210,249,674  (post-split prices)
Adjusted:    Close=$129.04, Vol=210,249,674  (already in post-split terms)
TotalReturn: Close=$517.09, Vol=52,562,418   (210M ÷ 4 = 52M in pre-split units)
```

**2020-09-02 (Post-Split)**

```
Unadjusted:  Close=$131.40, Vol=190,875,978
Adjusted:    Close=$131.40, Vol=190,875,978
TotalReturn: Close=$526.55, Vol=47,718,994
```

## ✅ Benefits of Golden Test

1. **Confidence**: Mathematical proof that refactoring is correct
1. **Regression Safety**: Catches accidental changes immediately
1. **Documentation**: Golden output IS the specification
1. **Fast Iteration**: Quick validation during development
1. **Clear Success**: Binary pass/fail, no ambiguity

## 🚀 Quick Start

### 1. Verify Golden Output Exists

```bash
ls -lh src/qtrader/1playground/golden_output.json
# Should show ~60KB file
```

### 2. View Golden Output Summary

```python
from test_validate_golden import load_golden_output
from pathlib import Path

golden = load_golden_output(Path("src/qtrader/1playground/golden_output.json"))
print(f"Symbol: {golden['metadata']['symbol']}")
print(f"Bars per series: {golden['series']['unadjusted']['bar_count']}")
```

### 3. After Refactoring

```python
# Your refactored code
from refactored_module import generate_canonical_series

# Generate test output with same structure as golden
test_data = generate_canonical_series()

# Validate
from test_validate_golden import validate_against_golden
passed = validate_against_golden(test_data, Path("golden_output.json"))

if passed:
    print("🎉 Refactoring successful!")
```

## 📊 Golden Output Structure

```json
{
  "metadata": {
    "symbol": "AAPL",
    "start_date": "2020-08-01",
    "end_date": "2020-09-03",
    "corporate_events": { ... }
  },
  "series": {
    "unadjusted": {
      "mode": "unadjusted",
      "bar_count": 24,
      "bars": [ ... ]
    },
    "adjusted": { ... },
    "total_return": { ... }
  }
}
```

## 🎓 Refactoring Tips

1. **Start Small**: Refactor one service at a time
1. **Test Often**: Run validation after each change
1. **Keep Original**: Don't delete current implementation until validation passes
1. **Match Structure**: Ensure test_data has same JSON structure as golden
1. **Debug with Key Points**: Use `validate_key_points()` for quick checks

## 📝 Next Steps

1. ✅ Golden output generated
1. ⏳ Refactor into service layers
1. ⏳ Create test harness for refactored code
1. ⏳ Run full validation
1. ⏳ Integrate into main qtrader package

______________________________________________________________________

**Status**: Golden output ready. Ready to begin refactoring.
