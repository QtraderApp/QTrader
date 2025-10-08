# QTrader Data Layer - Quick Reference

## 🚀 Quick Start

### Run All Tests

```bash
pytest tests/unit/models/ tests/integration/test_data_layer_corporate_events.py -v
```

### Generate Golden Output

```bash
python src/qtrader/1playground/generate_golden_output.py
```

### Validate Implementation

```bash
python src/qtrader/1playground/test_validation_script.py
```

## 📊 Test Coverage

**31 tests, 100% pass rate**

| Test File                             | Tests | Coverage            |
| ------------------------------------- | ----- | ------------------- |
| `test_canonical_bar.py`               | 12    | CanonicalBar models |
| `test_vendors_algoseek.py`            | 13    | Corporate events    |
| `test_data_layer_corporate_events.py` | 6     | Integration         |

## 🎯 Three Adjustment Modes

### 1. Unadjusted

- Raw as-traded prices
- Actual volume
- Dividend recorded on ex-date

### 2. Adjusted

- Split-adjusted prices (backward)
- Split-adjusted volume
- Dividend adjusted for splits

### 3. Total Return

- Forward compounding with dividend reinvestment
- Volume in starting-date units
- No dividend field (embedded in prices)

## 🐛 Bug Fixes Applied

### Dividend Extraction

```python
# CORRECT
return Decimal(str(self.AdjustmentFactor))
```

### Split Ratio Extraction

```python
# CORRECT - Algoseek stores inverse
split_ratio = Decimal("1") / Decimal(str(self.AdjustmentFactor))
```

**Algoseek format**:

- 4:1 split → AdjustmentFactor = 0.25
- 1:5 reverse split → AdjustmentFactor = 5.0

## 📈 Golden Output Results

**AAPL 2020-08-01 to 2020-09-01**

| Mode         | Return  | Interpretation    |
| ------------ | ------- | ----------------- |
| Unadjusted   | -69.21% | Split artifact    |
| Adjusted     | +23.17% | True price return |
| Total Return | +23.45% | With dividends    |

**Corporate events**:

- Dividend: $1.00 on 2020-08-07
- Split: 4:1 on 2020-08-31

**Validation**: 66 bars, 434 fields, 100% match

## 📁 Key Files

### Test Infrastructure

- `tests/unit/models/test_canonical_bar.py` - 270 lines, 12 tests
- `tests/unit/models/test_vendors_algoseek.py` - 560 lines, 13 tests
- `tests/integration/test_data_layer_corporate_events.py` - 520 lines, 6 tests

### Validation Infrastructure

- `src/qtrader/1playground/generate_golden_output.py` - Golden generator
- `src/qtrader/1playground/golden_output_new.json` - Reference output
- `src/qtrader/1playground/test_validate_golden.py` - Validation framework
- `src/qtrader/1playground/test_validation_script.py` - Test script

### Documentation

- `DATA_LAYER_VALIDATION_SUMMARY.md` - Complete summary
- `docs/DATA_LAYER_TEST_COVERAGE.md` - Test coverage report
- `src/qtrader/1playground/README.md` - Playground documentation

## 🔧 Common Tasks

### Change Test Symbol/Date Range

Edit `generate_golden_output.py`:

```python
SYMBOL = "AAPL"
START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
```

### Validate Refactoring

```python
from test_validate_golden import validate_against_golden

passed = validate_against_golden(test_data, Path("golden_output_new.json"))
```

### Check Corporate Events

```python
# Dividend detection
if bar.get_corporate_event_type() == CorporateEventType.CASH_DIVIDEND:
    amount = bar.get_dividend_amount()

# Split detection
if bar.get_corporate_event_type() in [CorporateEventType.FORWARD_SPLIT, ...]:
    ratio = bar.get_split_ratio()
```

## 📋 Testing Checklist

Before refactoring:

- [ ] Generate golden output with current implementation
- [ ] Verify golden output is correct (manual spot checks)
- [ ] Save golden output to version control

After refactoring:

- [ ] Run all unit tests
- [ ] Run integration tests
- [ ] Validate against golden output
- [ ] Check for regressions

## 🎓 Key Formulas

### Total Return

```
TR_t = TR_{t-1} × (UnAdj_t × Split_t + Div_t) / UnAdj_{t-1}
```

### Volume Adjustment (Total Return)

```
Volume_TR = Unadjusted_Volume / CumulativeSplitRatio
```

### Price Adjustment (Adjusted Mode)

```
Price_Adj = Unadjusted / (LastVolFactor / CurrentVolFactor)
```

## 🚨 Common Pitfalls

❌ **Using AdjustmentFactor directly for splits**

- Algoseek stores inverse: 0.25 for 4:1 split
- Always use: `1 / AdjustmentFactor`

❌ **Calculating dividend from close price**

- AdjustmentFactor contains dollar amount
- Use directly, no formula needed

❌ **Wrong volume units in Total Return**

- Should be in **starting-date units** (pre-split)
- Divide by cumulative split ratio, don't multiply

## 📞 Next Steps

### Phase 2: Adapter Layer

Update adapter to use new canonical models

### Phase 3: Downstream

Fix strategy interfaces and portfolio calculations

______________________________________________________________________

**Status**: ✅ Data layer validated and production-ready\
**Tests**: 31/31 passing (100%)\
**Golden Output**: 100% match (66 bars, 434 fields)
