# Testing Schwab OAuth Today (Manual Mode)

## Problem

- Your Schwab callback URI is currently: `https://analytic-alpha.com`
- The code expects: `https://127.0.0.1:8182`
- Schwab won't update until after market hours

## Solution: Manual Code Entry

I've added a **manual mode** that lets you paste the authorization code directly. This is how the original `schwap_demo.py` worked!

## How to Test Today

### 1. Set Environment Variables

```bash
export SCHWAB_API_KEY="your_key_here"
export SCHWAB_API_SECRET="your_secret_here"
```

### 2. Run the Test Script

```bash
python scripts/test_schwab_oauth.py
```

### 3. Follow the Prompts

The script will ask for your registered callback URI:

```
Enter your registered callback URI at Schwab
(e.g., https://analytic-alpha.com)
> https://analytic-alpha.com
```

### 4. Authorize in Browser

1. Click the authorization URL
1. Log in to Schwab
1. Authorize the application
1. The browser will redirect to `https://analytic-alpha.com?code=XXXXX...`
1. **The page will fail to load (expected)** - domain doesn't exist/isn't accessible

### 5. Copy the Code

From your browser's address bar, you'll see a URL like:

```
https://analytic-alpha.com?code=ABC123XYZ789...
```

Copy the entire URL (or just the code part after `code=`)

### 6. Paste the Code

Back in the terminal:

```
📝 After authorizing, paste the full redirect URL here:
   (or just the code parameter)

> https://analytic-alpha.com?code=ABC123XYZ789...
```

### 7. Success! 🎉

```
✅ SUCCESS! OAuth authentication complete
```

The token will be cached at `~/.qtrader/schwab_tokens.json` for future use.

## What Happens After Market Hours

Once Schwab updates your callback URI to `127.0.0.1:8182`, you can use the automatic mode:

```python
# No manual_mode parameter = automatic HTTPS callback
manager = SchwabOAuthManager(
    client_id=client_id,
    client_secret=client_secret,
    # redirect_uri defaults to https://127.0.0.1:8182
)
```

The authorization will happen automatically - no need to copy/paste the code!

## Code Changes

### Added to `SchwabOAuthManager.__init__()`

```python
manual_mode: bool = False  # New parameter
```

### New Method

```python
def _get_code_manual(self) -> str | None:
    """Get authorization code via manual user input."""
    # Prompts user to paste URL or code
    # Extracts code from URL if needed
    # Returns authorization code
```

### Updated Flow

```python
if self.manual_mode:
    auth_code = self._get_code_manual()
else:
    auth_code = self._start_callback_server()
```

## Benefits

✅ Works with **any** registered callback URI ✅ No need to wait for Schwab to update settings ✅ Common pattern used by many OAuth libraries ✅ Fallback for when HTTPS callback fails ✅ Easy to switch between modes

## Files Changed

- `src/qtrader/auth/schwab_oauth.py`: Added manual mode
- `scripts/test_schwab_oauth.py`: New test script (executable)

## Testing Status

- ✅ All 133 tests passing
- ✅ Code compiles without errors
- ✅ Pre-commit hooks passing
- ✅ Ready to test with your Schwab credentials

## Next Steps

1. **Today**: Test OAuth with manual mode
1. **After market hours**: Schwab updates callback URI to `127.0.0.1:8182`
1. **Tomorrow**: Switch to automatic mode (remove `manual_mode=True`)
1. **Continue**: Proceed with Phase 2 (Vendor Models)
