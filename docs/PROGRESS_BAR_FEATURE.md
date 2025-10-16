# Progress Bar Feature - Visual Guide

## Overview

The `qtrader data update` command now includes a **real-time progress bar** that shows:

- ✅ Which symbol is currently being processed
- ✅ Progress percentage (symbols completed / total)
- ✅ Progress bar visual indicator
- ✅ Time elapsed
- ✅ Success/failure status for each symbol

## Visual Examples

### Example 1: Updating 3 Symbols

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA,NVDA
```

**Live Progress Display:**

```
UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 3 symbols...

⠹ ✓ NVDA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 100% 0:00:08
```

**After Completion:**

```
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Status     ┃ Bars Added ┃ Date Range                ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-16  │
│ TSLA   │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-16  │
│ NVDA   │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-16  │
└────────┴────────────┴────────────┴───────────────────────────┘

Successful: 3/3
Total bars added: 15
```

### Example 2: Updating Many Symbols (50+)

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

**Live Progress Display:**

```
UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 50 cached symbols...

⠙ ✓ IBM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23/50 46% 0:01:15
```

The progress bar shows:

- **⠙** - Spinner (animated, shows activity)
- **✓ IBM** - Currently processing IBM (✓ = success, ✗ = error)
- **━━━━━━━** - Visual progress bar
- **23/50** - 23 symbols completed out of 50 total
- **46%** - Percentage complete
- **0:01:15** - Time elapsed (1 minute 15 seconds)

### Example 3: With Errors

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,INVALID,TSLA
```

**Live Progress Display:**

```
UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 3 symbols...

⠹ ✗ INVALID ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/3 67% 0:00:05
```

Notice the **✗** instead of **✓** when a symbol fails.

**After Completion:**

```
┏━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol  ┃ Status     ┃ Bars Added ┃ Date Range                ┃
┡━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL    │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-16  │
│ INVALID │ ✗ Error    │          - │ -                         │
│ TSLA    │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-16  │
└─────────┴────────────┴────────────┴───────────────────────────┘

Successful: 2/3
Total bars added: 10

Errors (1):
  • INVALID: Symbol not found in Schwab API
```

### Example 4: Dry Run with Progress

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run
```

**Live Progress Display:**

```
DRY RUN Dataset: schwab-us-equity-1d-adjusted

Updating 10 cached symbols...

⠸ ✓ MSFT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7/10 70% 0:00:02
```

Dry run is **fast** because it doesn't make API calls - just checks metadata.

### Example 5: All Symbols Already Current

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

**Live Progress Display:**

```
UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 5 cached symbols...

⠹ ✓ GOOGL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5/5 100% 0:00:03
```

**After Completion:**

```
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Symbol ┃ Status     ┃ Bars Added ┃ Date Range ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ AAPL   │ ✓ Current  │          - │ -          │
│ TSLA   │ ✓ Current  │          - │ -          │
│ NVDA   │ ✓ Current  │          - │ -          │
│ GOOGL  │ ✓ Current  │          - │ -          │
│ MSFT   │ ✓ Current  │          - │ -          │
└────────┴────────────┴────────────┴────────────┘

Successful: 5/5
Total bars added: 0
```

All symbols already up-to-date (market not open yet or already ran update today).

## Progress Bar Components

### 1. Spinner

```
⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
```

Animated spinner rotates to show activity (process is running)

### 2. Status Symbol

- **✓** - Symbol updated successfully
- **✗** - Symbol failed with error

### 3. Current Symbol

Shows which symbol is being processed right now

### 4. Progress Bar

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Visual bar that fills up as symbols are processed

### 5. Counters

```
23/50
```

Shows: `{completed}/{total}`

### 6. Percentage

```
46%
```

Percentage of symbols completed

### 7. Elapsed Time

```
0:01:15
```

Time elapsed since start (minutes:seconds)

## Benefits

### Real-Time Feedback

- **See progress instantly** instead of waiting for completion
- **Know which symbol is being processed** (useful for debugging)
- **Estimate time remaining** (can calculate: elapsed / percentage)

### User Experience

- **Professional appearance** with Rich formatting
- **Non-blocking** - doesn't freeze terminal
- **Informative** - shows success/failure per symbol in real-time

### Performance Monitoring

- **Track processing speed** - see if some symbols are slower
- **Identify problems early** - see errors as they happen
- **Time your updates** - know how long batch updates take

## Technical Details

### Implementation

- Uses **Rich Progress** library
- **Streaming results** from DatasetUpdater iterator
- **Live updates** - progress bar refreshes as each symbol completes
- **Non-buffering** - shows results immediately

### Customization

Progress bar can be configured in `src/qtrader/cli.py`:

```python
with Progress(
    SpinnerColumn(),              # Animated spinner
    "[progress.description]{task.description}",  # Custom text
    BarColumn(),                  # Visual progress bar
    TaskProgressColumn(),         # "23/50 46%"
    TimeElapsedColumn(),          # "0:01:15"
    console=console,
) as progress:
    ...
```

### Performance Impact

- **Minimal overhead** - progress updates are very fast
- **Efficient streaming** - processes symbols one at a time
- **No memory bloat** - doesn't load all results upfront

## Comparison: Before vs After

### Before (No Progress Bar)

```bash
$ qtrader data update --dataset schwab-us-equity-1d-adjusted

UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 50 cached symbols...

[Wait 2-3 minutes with no feedback]

┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Status     ┃ Bars Added ┃ Date Range        ┃
...
```

**Issues:**

- ❌ No indication of progress
- ❌ Don't know if it's frozen or working
- ❌ Can't estimate completion time
- ❌ No early error detection

### After (With Progress Bar)

```bash
$ qtrader data update --dataset schwab-us-equity-1d-adjusted

UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 50 cached symbols...

⠙ ✓ IBM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23/50 46% 0:01:15

[Summary table appears when done]
```

**Benefits:**

- ✅ See real-time progress
- ✅ Know exactly which symbol is processing
- ✅ Estimate time remaining
- ✅ Catch errors immediately

## Usage Tips

### 1. Monitor Large Updates

For 100+ symbols, the progress bar helps you:

- See if it's still working
- Estimate how much longer
- Identify slow symbols

### 2. Debugging

If a symbol fails:

- You see it immediately (✗ symbol name)
- Don't have to wait for all symbols to finish
- Can cancel and fix the issue

### 3. Performance Testing

Compare elapsed times:

- With cache: ~1-2 seconds per symbol
- Without cache: ~5-10 seconds per symbol
- Can verify smart caching is working!

### 4. Automation Scripts

When running in scripts:

- Progress bar still works in cron jobs
- Can redirect to log file
- Time stamps help with scheduling

## Related Features

- **Verbose Mode:** `--verbose` adds detailed logging alongside progress bar
- **Dry Run:** `--dry-run` shows progress without making API calls (very fast)
- **Symbol Filter:** `--symbols` limits progress to specific symbols

## Future Enhancements

Potential improvements:

- [ ] Show **estimated time remaining** (ETA)
- [ ] **Parallel processing** with multiple progress bars
- [ ] **Rate limit indicator** (show when waiting for rate limiter)
- [ ] **Bandwidth usage** (show KB/MB transferred)
- [ ] **API call counter** (show calls made / remaining)

______________________________________________________________________

**The progress bar makes data updates transparent and professional!** 🚀
