# Schwab API Demo

Comprehensive script demonstrating how to use Schwab's APIs:

- **Historical Data**: Fetch past price data via REST API
- **Streaming Data**: Receive real-time candles via WebSocket

## Prerequisites

### 1. Register Your Application

1. Go to [Schwab Developer Portal](https://developer.schwab.com/)
1. Create a new app to get your API Key and Secret
1. Set your redirect URI to: `https://127.0.0.1:8080`

### 2. Install Dependencies

```bash
pip install requests websockets
```

### 3. Set Environment Variables

**Default mode (Manual Code Entry) - Works with ANY redirect URI:**

```bash
export SCHWAB_API_KEY="your_api_key_here"
export SCHWAB_API_SECRET="your_api_secret_here"
export SCHWAB_REDIRECT_URI="https://alpha-q.com"  # Your registered redirect URI
export SCHWAB_MANUAL_CODE="true"  # Use manual code entry
```

**Alternative mode (Automatic Callback Server) - Only works if redirect URI is <https://127.0.0.1:8080>:**

```bash
export SCHWAB_API_KEY="your_api_key_here"
export SCHWAB_API_SECRET="your_api_secret_here"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8080"
export SCHWAB_MANUAL_CODE="false"  # Use callback server
```

Or create a `.env` file (not tracked by git):

```bash
SCHWAB_API_KEY=your_api_key_here
SCHWAB_API_SECRET=your_api_secret_here
SCHWAB_REDIRECT_URI=https://alpha-q.com
SCHWAB_MANUAL_CODE=true
```

## Usage

### Interactive Mode (Recommended)

```bash
python schwap_demo.py
```

You'll be prompted to choose:

1. **Historical data** - Fetch past candles (daily, minute, etc.)
1. **Streaming data** - Live 1-minute candles

### Non-Interactive Mode

Set the mode via environment variable:

```bash
# Historical mode
export SCHWAB_MODE="historical"
python schwap_demo.py

# Streaming mode
export SCHWAB_MODE="streaming"
python schwap_demo.py
```

## What It Does

### Mode 1: Historical Data (REST API)

Fetches historical price data with customizable parameters:

- **Symbols**: Any equity (AAPL, TSLA, GOOGL, etc.)
- **Period Types**: day, month, year, ytd
- **Frequencies**: minute (1, 5, 10, 15, 30), daily, weekly, monthly
- **Periods**: 1-N periods

**Example Session:**

```
Symbol: AAPL
Period Type: month (last month)
Period: 1
Frequency Type: daily
Frequency: 1

Shows: Last month of daily OHLCV candles with summary statistics
```

**Use Cases:**

- Backtesting strategies
- Historical analysis
- Data collection
- Performance attribution

### Mode 2: Streaming Data (WebSocket)

Connects to Schwab's streaming server for real-time data:

- **Real-time**: 1-minute candles as they form
- **Live Quotes**: During market hours only
- **WebSocket**: Persistent connection

**Example Session:**

```
Connected to Schwab Streamer
LOGIN successful!
Subscribing to CHART_EQUITY for AAPL...

--- Chart Data ---
Symbol: AAPL
  Open:   150.25
  High:   150.89
  Low:    150.10
  Close:  150.75
  Volume: 125430
  Time:   1696950600000
```

**Use Cases:**

- Live trading
- Real-time monitoring
- Alert systems
- Market watching

## Output Format

### Historical Data Output

Displays a formatted table with:

- **Date/Time**: Timestamp for the candle
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume

Plus summary statistics:

- Period change ($ and %)
- High/Low for entire period
- Total volume

### Streaming Data Output

Each candle includes:

- **Symbol**: Ticker (AAPL)
- **Open**: Opening price for the minute
- **High**: Highest price for the minute
- **Low**: Lowest price for the minute
- **Close**: Closing price for the minute
- **Volume**: Total volume for the minute
- **Time**: Timestamp in milliseconds since epoch

## Notes

- The script runs a local HTTP server on port 8080 to receive the OAuth callback
- Press `Ctrl+C` to stop streaming and logout gracefully
- Access tokens expire after a certain period (check `expires_in` in the response)
- Chart data is only available during market hours

## Troubleshooting

**Old redirect URI / No access to domain:**

Use manual code mode (default):

```bash
export SCHWAB_REDIRECT_URI="https://your-old-domain.com"
export SCHWAB_MANUAL_CODE="true"
```

After authorization, Schwab will try to redirect to the old domain (which will fail), but you can copy the code from the URL in your browser:

```
https://your-old-domain.com?code=COPY_THIS_CODE&session=...
```

**Port 8080 already in use (automatic mode only):**

- Change `REDIRECT_URI` in the script to use a different port
- Update your app settings in Schwab Developer Portal to match

**Authorization failed:**

- Verify your API Key and Secret are correct
- Ensure redirect URI matches exactly what's registered in Schwab
- Check your app has the necessary permissions

**No data streaming:**

- Chart data only streams during market hours (9:30 AM - 4:00 PM ET)
- Try during regular trading hours

## API Documentation

See `docs/schwap/api.md` for complete Schwab Streamer API documentation.
