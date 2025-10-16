# DataConfig Usage Audit

**Date:** October 16, 2025 **Context:** Sprint 1 - Pre-DataSourceSelector integration **Purpose:** Identify unused fields before refactoring

______________________________________________________________________

## Summary

| Component                  | Status    | Usage Count | Notes                           |
| -------------------------- | --------- | ----------- | ------------------------------- |
| **ValidationConfig**       | ✅ Active | High        | Used in tests, DataConfig       |
| **BarSchemaConfig**        | ✅ Active | High        | Required field, used everywhere |
| **AdjustmentSchemaConfig** | ✅ Active | Medium      | Optional, used in tests         |
| **DataConfig**             | ✅ Active | High        | Core config class               |
| **strict_frequency**       | ⚠️ UNUSED | 0           | Only in tests, never checked    |
| **decimals**               | ⚠️ UNUSED | 0           | Only in tests, never checked    |
| **adjustment_schema**      | ✅ Active | Low         | Optional, has tests             |

______________________________________________________________________

## Detailed Findings

### ✅ **Active Components**

#### 1. ValidationConfig

- **Purpose:** OHLC validation rules
- **Fields:** `epsilon`, `ohlc_policy`, `close_only_fields`
- **Usage:**
  - ✅ Used in `DataConfig` (default_factory)
  - ✅ Tested in `test_data_config.py`
  - ✅ Referenced in specs (phase01.md)
- **Verdict:** Keep - Core validation config

#### 2. BarSchemaConfig

- **Purpose:** Map vendor columns to canonical Bar fields
- **Fields:** `ts`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- **Usage:**
  - ✅ Required field in `DataConfig`
  - ✅ Used in examples (`data_service_example.py`)
  - ✅ Used in tests (`test_data_service.py`)
  - ✅ Adapter integration point
- **Verdict:** Keep - Essential for vendor integration

#### 3. AdjustmentSchemaConfig

- **Purpose:** Map vendor columns to AdjustmentEvent fields (optional)
- **Fields:** `ts`, `symbol`, `event_type`, `px_factor`, `vol_factor`, `metadata_fields`
- **Usage:**
  - ✅ Optional field in `DataConfig`
  - ✅ Tested in `test_data_config.py`
  - ✅ Part of adjustment events system
- **Verdict:** Keep - Used for corporate actions

#### 4. DataConfig (core fields)

- `mode`: ✅ Used (adjustment mode selection)
- `frequency`: ✅ Used (bar frequency)
- `timezone`: ✅ Used (timestamp normalization)
- `source_tag`: ⚠️ **TO BE REPLACED** by `source_selector` (Sprint 1 Day 3)
- `validation`: ✅ Used (ValidationConfig)
- `bar_schema`: ✅ Used (BarSchemaConfig - required)
- `adjustment_schema`: ✅ Used (AdjustmentSchemaConfig - optional)

______________________________________________________________________

### ⚠️ **UNUSED Fields** (Candidates for Removal)

#### 1. strict_frequency

```python
strict_frequency: bool = Field(default=True, description="Raise on frequency mismatch")
```

**Search Results:**

- ❌ NOT used in any `src/` code
- ✅ Only in test: `test_data_config.py` (checks default value)
- ✅ Mentioned in docs: `phase01.md`, YAML examples

**Code Search:**

```bash
grep -r "strict_frequency" src/
# NO RESULTS in source code
```

**Recommendation:**

- **REMOVE** from `DataConfig` (not implemented)
- **ALTERNATIVE:** Add to future roadmap if frequency validation is needed
- **MIGRATION:** No impact (never used in code)

#### 2. decimals

```python
decimals: dict[str, int] = Field(default={"price": 4, "cash": 4}, description="Decimal precision")
```

**Search Results:**

- ❌ NOT used in any `src/` code (except `system_config.py` has `price_decimals`)
- ✅ Only in test: `test_data_config.py` (checks default value)
- ✅ Mentioned in docs: `phase01.md`, YAML examples

**Code Search:**

```bash
grep -r "config.decimals" src/
# NO RESULTS - field never accessed
```

**Note:** `system_config.py` has separate `price_decimals` field (not same as `DataConfig.decimals`)

**Recommendation:**

- **REMOVE** from `DataConfig` (not implemented)
- **ALTERNATIVE:** Use `system_config.DataConfig.price_decimals` if needed
- **MIGRATION:** No impact (never used in code)

______________________________________________________________________

## Recommendations

### Immediate Actions (Sprint 1 Day 3)

1. **Update DataConfig:**

   ```python
   class DataConfig(BaseModel):
       """Data loading and processing configuration."""

       mode: str = Field(default="adjusted", ...)
       frequency: str = Field(default="1d", ...)
       timezone: str = Field(default="America/New_York", ...)

       # NEW: Replace source_tag with source_selector
       source_selector: DataSourceSelector = Field(...)

       # Keep these (actively used)
       validation: ValidationConfig = Field(...)
       bar_schema: BarSchemaConfig = Field(...)
       adjustment_schema: Optional[AdjustmentSchemaConfig] = Field(...)

       # REMOVE: Never implemented
       # strict_frequency: bool = Field(...)  ❌
       # decimals: dict[str, int] = Field(...)  ❌
   ```

1. **Update Tests:**

   - Remove assertions for `strict_frequency`
   - Remove assertions for `decimals`
   - Add tests for `source_selector`

1. **Update Docs:**

   - Remove `strict_frequency` from YAML examples
   - Remove `decimals` from YAML examples (or clarify it's in `system_config`)
   - Add `source_selector` examples

______________________________________________________________________

## Duplicate Config Classes

**Warning:** There are TWO different `DataConfig` classes:

1. **`src/qtrader/config/data_config.py`** (this file)

   - Pydantic BaseModel
   - Used by DataService
   - Fields: mode, frequency, timezone, source_tag, bar_schema, etc.

1. **`src/qtrader/config/system_config.py`**

   - Python dataclass
   - Used by SystemConfig
   - Fields: sources_config, default_mode, price_decimals, cache_enabled, etc.

**These are DIFFERENT classes with DIFFERENT purposes:**

- `data_config.DataConfig`: **Per-request data loading config** (what data to load, how to parse)
- `system_config.DataConfig`: **System-wide data settings** (caching, validation, defaults)

**Recommendation:** Consider renaming one to avoid confusion:

- Rename `data_config.DataConfig` → `DataLoadConfig` or `DataRequestConfig`
- OR rename `system_config.DataConfig` → `DataSystemConfig`

______________________________________________________________________

## Migration Impact

### Breaking Changes

- ✅ `source_tag` → `source_selector`: **Expected (Sprint 1 goal)**
- ⚠️ `strict_frequency` removal: **Non-breaking** (never used)
- ⚠️ `decimals` removal: **Non-breaking** (never used)

### Test Updates Required

- `tests/unit/config/test_data_config.py`: Remove unused field assertions
- All tests using `DataConfig`: Add `source_selector`
- Examples: Update to use `source_selector`

______________________________________________________________________

## Conclusion

**Keep separate files:**

- ✅ `data_config.py`: Config schemas (ValidationConfig, BarSchemaConfig, etc.)
- ✅ `data_source_selector.py`: Business logic (AssetClass, DataType, DataSourceSelector)

**Clean up unused fields:**

- ❌ Remove `strict_frequency` (not implemented)
- ❌ Remove `decimals` (not implemented, use `system_config` version if needed)
- ✅ Keep all other fields (actively used)

**Sprint 1 Day 3:**

- Update `DataConfig` to use `source_selector`
- Remove unused fields
- Update all tests and examples
