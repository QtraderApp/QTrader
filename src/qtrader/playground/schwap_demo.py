"""Simple script to get AAPL data from Schwab API.

This demonstrates:
1. OAuth authentication to get access token
2. Token caching (reuse tokens until expired)
3. Two modes:
   a) Historical: Fetch historical price data via REST API
   b) Streaming: Stream real-time candles via WebSocket
"""

import asyncio
import base64
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import requests
import websockets

# Get credentials from environment
API_KEY = os.environ.get("SCHWAB_API_KEY")
API_SECRET = os.environ.get("SCHWAB_API_SECRET")
REDIRECT_URI = os.environ.get("SCHWAB_REDIRECT_URI")  # Your registered redirect URI
USE_MANUAL_CODE = (
    os.environ.get("SCHWAB_MANUAL_CODE", "true").lower() == "true"
)  # Set to "false" to use callback server

# Token cache file
TOKEN_CACHE_FILE = Path.home() / ".schwab_token_cache.json"

if not API_KEY or not API_SECRET or not REDIRECT_URI:
    raise ValueError(
        "Missing required environment variables:\n"
        "  SCHWAB_API_KEY (your app key)\n"
        "  SCHWAB_API_SECRET (your app secret)\n"
        "  SCHWAB_REDIRECT_URI (your redirect URI)"
    )

# OAuth endpoints
AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
API_BASE = "https://api.schwabapi.com"

# Global to capture authorization code
auth_code = None


class CallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback."""

    def do_GET(self) -> None:
        """Capture authorization code from callback."""
        global auth_code
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if "code" in params:
            auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>"
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:
        """Suppress log messages."""
        pass


def load_cached_token() -> dict | None:
    """Load cached token from file if it exists and is valid.

    Returns:
        Token data dict if valid, None otherwise
    """
    if not TOKEN_CACHE_FILE.exists():
        return None

    try:
        with open(TOKEN_CACHE_FILE) as f:
            token_data: dict[str, object] = json.load(f)

        # Check if token is expired
        expires_at = token_data.get("expires_at", 0)
        if time.time() < expires_at:  # type: ignore[operator]
            remaining = int((expires_at - time.time()) / 60)  # type: ignore[operator]
            print(f"✓ Using cached token (expires in {remaining} minutes)\n")
            return token_data
        else:
            print("⚠ Cached token expired\n")
            return None

    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None


def save_token_cache(token_data: dict) -> None:
    """Save token to cache file.

    Args:
        token_data: Token response from Schwab API
    """
    # Calculate expiration time (expires_in is in seconds)
    expires_in = token_data.get("expires_in", 1800)  # Default 30 min
    cache_data = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "expires_in": expires_in,
        "expires_at": time.time() + expires_in,
        "token_type": token_data.get("token_type", "Bearer"),
    }

    with open(TOKEN_CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

    # Set restrictive permissions (owner only)
    TOKEN_CACHE_FILE.chmod(0o600)

    print(f"✓ Token cached to {TOKEN_CACHE_FILE}")
    print(f"  (valid for {expires_in // 60} minutes)\n")


def get_access_token(force_new: bool = False) -> str:
    """Get access token via OAuth flow or from cache.

    Args:
        force_new: Force new authentication even if cached token exists

    Returns:
        Access token string
    """
    # Try to use cached token first
    if not force_new:
        cached = load_cached_token()
        if cached:
            return str(cached["access_token"])

    print("No valid cached token - starting OAuth flow...")

    global auth_code

    # Step 1: Build authorization URL
    auth_params = {
        "client_id": API_KEY,
        "redirect_uri": REDIRECT_URI,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    print("\n" + "=" * 70)
    print("AUTHORIZATION REQUIRED")
    print("=" * 70)
    print(f"\nPlease visit this URL to authorize:\n\n{auth_url}\n")

    if USE_MANUAL_CODE:
        # Manual mode: User copies the code from the redirect URL
        print("=" * 70)
        print("MANUAL MODE")
        print("=" * 70)
        print("\nSteps:")
        print("1. Click the authorization URL above")
        print("2. Log in and authorize the application")
        print(f"3. You'll be redirected to: {REDIRECT_URI}?code=...")
        print("4. Copy the 'code' parameter from the URL")
        print("5. Paste it below")
        print("\nExample URL after redirect:")
        print(f"{REDIRECT_URI}?code=ABC123XYZ&session=...")
        print("                     ^^^^^^^^^")
        print("                   Copy this part")
        print("\nTIP: You can paste the entire URL or just the code!\n")

        user_input = input("Paste the authorization code (or full URL): ").strip()

        if not user_input:
            raise RuntimeError("No authorization code provided")

        # Check if user pasted the full URL instead of just the code
        if "code=" in user_input:
            # Extract code from URL
            parsed = urllib.parse.urlparse(user_input)
            params = urllib.parse.parse_qs(parsed.query)
            if "code" in params:
                auth_code = params["code"][0]
                print("✓ Extracted code from URL")
            else:
                raise RuntimeError("Could not find 'code' parameter in URL")
        else:
            # User pasted just the code - URL decode it in case it's encoded
            auth_code = urllib.parse.unquote(user_input)

        print(f"Code length: {len(auth_code)} characters")
        print("✓ Received authorization code\n")
    else:
        # Automatic mode: Use callback server
        print("Waiting for callback...")

        # Step 2: Start local server to receive callback
        server = HTTPServer(("127.0.0.1", 8080), CallbackHandler)
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # Wait for authorization code
        server_thread.join(timeout=120)  # 2 minute timeout

        if not auth_code:
            raise RuntimeError("Failed to get authorization code")

        print("✓ Received authorization code\n")

    # Step 3: Exchange code for access token
    print("Exchanging code for access token...")

    credentials = f"{API_KEY}:{API_SECRET}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
    }

    response = None
    try:
        response = requests.post(TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print("\n❌ Token exchange failed!")
        if response:
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
        print("\nDebug Info:")
        print(f"  Redirect URI used: {REDIRECT_URI}")
        print(f"  Code length: {len(auth_code)}")
        print(f"  Code sample: {auth_code[:20]}...")
        print("\nCommon issues:")
        print("  1. Authorization code already used (codes are single-use)")
        print("  2. Redirect URI mismatch (must exactly match registered URI)")
        print("  3. Code expired (codes expire in 30-60 seconds)")
        print("  4. Invalid API credentials")
        print("\nTry running the script again and:")
        print("  - Be faster when copying the code")
        print("  - Make sure redirect URI exactly matches Schwab Developer Portal")
        print("  - Verify your API key and secret are correct")
        raise

    token_data = response.json()
    access_token = str(token_data["access_token"])

    print(f"✓ Got access token (expires in {token_data.get('expires_in', 'unknown')}s)")

    # Save to cache for future use
    save_token_cache(token_data)

    return access_token


def get_user_preferences(access_token: str) -> dict:
    """Get user preferences including streamer info.

    Note: This endpoint requires 'Account and Trading Production' OAuth scope.
    If your app doesn't have this scope, you'll get a 401 error.
    """
    print("Fetching user preferences...")

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(f"{API_BASE}/trader/v1/userPreference", headers=headers)
        response.raise_for_status()
        preferences: dict = response.json()
        print("✓ Got user preferences\n")
        return preferences
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("\n" + "=" * 70)
            print("❌ ERROR: Unable to fetch user preferences (401 Unauthorized)")
            print("=" * 70)
            print("\nYour app is missing required OAuth scopes.")
            print("\nTo fix this:")
            print("1. Go to: https://developer.schwab.com/")
            print("2. Open your app settings")
            print("3. Under 'OAuth Scopes', enable:")
            print("   ✓ Market Data")
            print("   ✓ Account and Trading Production")
            print("4. Save changes")
            print("5. Clear token cache: python schwap_demo.py --clear-cache")
            print("6. Run again and re-authorize")
            print("\nNote: Without proper scopes, streaming mode won't work.")
            print("      Historical mode should still work with 'Market Data' scope only.")
            print("=" * 70)
            raise RuntimeError("Missing required OAuth scopes for streaming")
        else:
            raise


def get_historical_data(
    access_token: str,
    symbol: str = "AAPL",
    period_type: str = "month",
    period: int = 1,
    frequency_type: str = "daily",
    frequency: int = 1,
) -> dict:
    """Get historical price data from Schwab REST API.

    Args:
        access_token: OAuth access token
        symbol: Stock symbol (default: AAPL)
        period_type: Type of period - day, month, year, ytd (default: month)
        period: Number of periods (default: 1)
        frequency_type: Frequency type - minute, daily, weekly, monthly (default: daily)
        frequency: Frequency value - 1, 5, 10, 15, 30 for minute; 1 for others (default: 1)

    Returns:
        Dictionary containing candles data

    Examples:
        # Last month of daily candles
        get_historical_data(token, "AAPL", "month", 1, "daily", 1)

        # Last 10 days of 5-minute candles
        get_historical_data(token, "AAPL", "day", 10, "minute", 5)

        # Year-to-date weekly candles
        get_historical_data(token, "AAPL", "ytd", 1, "weekly", 1)
    """
    print(f"Fetching historical data for {symbol}...")
    print(f"  Period: {period} {period_type}(s)")
    print(f"  Frequency: {frequency} {frequency_type}")

    headers = {"Authorization": f"Bearer {access_token}"}
    params: dict[str, str | int] = {
        "symbol": symbol,
        "periodType": period_type,
        "period": period,
        "frequencyType": frequency_type,
        "frequency": frequency,
    }

    try:
        response = requests.get(
            f"{API_BASE}/marketdata/v1/pricehistory",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("\n" + "=" * 70)
            print("❌ ERROR: Unable to fetch historical data (401 Unauthorized)")
            print("=" * 70)
            print("\nYour app is missing required OAuth scope.")
            print("\nTo fix this:")
            print("1. Go to: https://developer.schwab.com/")
            print("2. Open your app settings")
            print("3. Under 'OAuth Scopes', enable:")
            print("   ✓ Market Data")
            print("4. Save changes")
            print("5. Clear token cache: python schwap_demo.py --clear-cache")
            print("6. Run again and re-authorize")
            print("=" * 70)
            raise RuntimeError("Missing required OAuth scope for market data")
        else:
            print(f"\nError fetching data: {e}")
            print(f"Status: {e.response.status_code}")
            print(f"Response: {e.response.text}")
            raise

    data: dict = response.json()
    candles = data.get("candles", [])

    print(f"✓ Received {len(candles)} candles\n")

    return data


def check_data_availability(
    access_token: str,
    symbol: str = "AAPL",
    frequency_type: str = "daily",
) -> dict:
    """Check how far back historical data is available for a symbol.

    Args:
        access_token: OAuth access token
        symbol: Stock symbol (default: AAPL)
        frequency_type: Frequency type - minute, daily, weekly, monthly (default: daily)

    Returns:
        Dictionary containing data availability information
    """
    print(f"\nChecking data availability for {symbol} ({frequency_type})...")

    # Request maximum period to see how far back data goes
    period_map = {
        "minute": ("day", 10),  # Max 10 days for minute data
        "daily": ("year", 20),  # Max 20 years for daily data
        "weekly": ("year", 20),  # Max 20 years for weekly data
        "monthly": ("year", 20),  # Max 20 years for monthly data
    }

    period_type, max_period = period_map.get(frequency_type, ("year", 20))

    try:
        data = get_historical_data(
            access_token,
            symbol=symbol,
            period_type=period_type,
            period=max_period,
            frequency_type=frequency_type,
            frequency=1,
        )

        candles = data.get("candles", [])

        from datetime import datetime

        # Display results header
        print("\n" + "=" * 70)
        print(f"Data Availability Report: {symbol}")
        print("=" * 70)
        print(f"\nFrequency Type: {frequency_type}")

        if not candles:
            # No data available - show N/A for all fields
            result = {
                "symbol": symbol,
                "frequency_type": frequency_type,
                "available": False,
                "first_date": None,
                "last_date": None,
                "total_candles": 0,
                "days_available": 0,
                "years_available": 0,
            }

            print("First Available: N/A")
            print("Last Available:  N/A")
            print("Total Candles:   0")
            print("Days of Data:    0")
            print("Years of Data:   0.00")
            print("\n⚠ No data available (symbol may be delisted or invalid)")
            print("=" * 70 + "\n")

            return result

        # Data available - calculate statistics
        first_candle = candles[0]
        last_candle = candles[-1]

        first_date = datetime.fromtimestamp(first_candle["datetime"] / 1000)
        last_date = datetime.fromtimestamp(last_candle["datetime"] / 1000)

        days_available = (last_date - first_date).days
        years_available = days_available / 365.25

        result = {
            "symbol": symbol,
            "frequency_type": frequency_type,
            "available": True,
            "first_date": first_date.strftime("%Y-%m-%d"),
            "last_date": last_date.strftime("%Y-%m-%d"),
            "total_candles": len(candles),
            "days_available": days_available,
            "years_available": round(years_available, 2),
        }

        print(f"First Available: {result['first_date']}")
        print(f"Last Available:  {result['last_date']}")
        print(f"Total Candles:   {result['total_candles']:,}")
        print(f"Days of Data:    {result['days_available']:,}")
        print(f"Years of Data:   {result['years_available']:.2f}")
        print("\n" + "=" * 70 + "\n")

        return result

    except Exception as e:
        print(f"\n❌ Error checking data availability: {e}\n")
        return {
            "symbol": symbol,
            "frequency_type": frequency_type,
            "available": False,
            "error": str(e),
        }


def display_historical_candles(data: dict, limit: int | None = 10) -> None:
    """Display historical candles in a readable format.

    Args:
        data: Data returned from get_historical_data
        limit: Maximum number of candles to display (default: 10, None for all)
    """
    candles = data.get("candles", [])
    symbol = data.get("symbol", "UNKNOWN")

    if not candles:
        print("No candles found!")
        return

    print("=" * 80)
    print(f"Historical Data for {symbol}")
    print("=" * 80)

    # Display settings
    display_candles = candles[-limit:] if limit else candles

    print(f"\nShowing {len(display_candles)} of {len(candles)} candles:\n")

    # Header
    print(f"{'Date/Time':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12}")
    print("-" * 80)

    # Candles
    for candle in display_candles:
        timestamp = candle.get("datetime", 0)
        # Convert milliseconds to readable format
        from datetime import datetime

        dt = datetime.fromtimestamp(timestamp / 1000)
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        open_price = candle.get("open", 0)
        high_price = candle.get("high", 0)
        low_price = candle.get("low", 0)
        close_price = candle.get("close", 0)
        volume = candle.get("volume", 0)

        print(
            f"{date_str:<20} {open_price:>10.2f} {high_price:>10.2f} "
            f"{low_price:>10.2f} {close_price:>10.2f} {volume:>12,}"
        )

    print("\n" + "=" * 80)

    # Summary statistics
    if candles:
        first_candle = candles[0]
        last_candle = candles[-1]
        first_close = first_candle.get("close", 0)
        last_close = last_candle.get("close", 0)
        change = last_close - first_close
        change_pct = (change / first_close * 100) if first_close else 0

        print("\nSummary:")
        print(f"  First Close: ${first_close:.2f}")
        print(f"  Last Close:  ${last_close:.2f}")
        print(f"  Change:      ${change:+.2f} ({change_pct:+.2f}%)")
        print(f"  High:        ${max(c.get('high', 0) for c in candles):.2f}")
        print(f"  Low:         ${min(c.get('low', 0) for c in candles):.2f}")
        total_volume = sum(c.get("volume", 0) for c in candles)
        print(f"  Total Vol:   {total_volume:,}")
    print()


async def stream_chart_data(access_token: str, streamer_url: str, customer_id: str, correl_id: str) -> None:
    """Connect to Schwab WebSocket and stream AAPL chart candles.

    Args:
        access_token: OAuth access token
        streamer_url: WebSocket URL from user preferences
        customer_id: Customer ID from user preferences
        correl_id: Correlation ID from user preferences
    """
    async with websockets.connect(streamer_url) as websocket:
        print("Connected to Schwab Streamer")

        # Step 1: LOGIN
        login_request = {
            "requests": [
                {
                    "requestid": "1",
                    "service": "ADMIN",
                    "command": "LOGIN",
                    "SchwabClientCustomerId": customer_id,
                    "SchwabClientCorrelId": correl_id,
                    "parameters": {
                        "Authorization": access_token,
                        "SchwabClientChannel": "N9",
                        "SchwabClientFunctionId": "APIAPP",
                    },
                }
            ]
        }

        print("\nSending LOGIN request...")
        await websocket.send(json.dumps(login_request))

        # Wait for login response
        response = await websocket.recv()
        login_response = json.loads(response)
        print(f"LOGIN response: {json.dumps(login_response, indent=2)}")

        # Check if login was successful
        if login_response.get("response", [{}])[0].get("content", {}).get("code") != 0:
            print("LOGIN failed!")
            return

        print("\nLOGIN successful!")

        # Step 2: Subscribe to CHART_EQUITY for AAPL
        chart_request = {
            "requests": [
                {
                    "requestid": "2",
                    "service": "CHART_EQUITY",
                    "command": "SUBS",
                    "SchwabClientCustomerId": customer_id,
                    "SchwabClientCorrelId": correl_id,
                    "parameters": {
                        "keys": "AAPL",
                        "fields": "0,1,2,3,4,5,6,7,8",  # All chart fields
                    },
                }
            ]
        }

        print("\nSubscribing to CHART_EQUITY for AAPL...")
        await websocket.send(json.dumps(chart_request))

        # Wait for subscription response
        response = await websocket.recv()
        sub_response = json.loads(response)
        print(f"Subscription response: {json.dumps(sub_response, indent=2)}")

        # Step 3: Listen for streaming data
        print("\nListening for chart data (press Ctrl+C to stop)...")
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)

                # Pretty print the data
                if "data" in data:
                    print("\n--- Chart Data ---")

                    # Capture reception time (when we received the message)
                    import time

                    reception_time = time.time()  # Current time in seconds since epoch

                    for item in data["data"]:
                        if item.get("service") == "CHART_EQUITY":
                            for content in item.get("content", []):
                                # Note: API returns fields in different order than documented!
                                # Based on actual response analysis:
                                # '1' = Volume (in lots/hundreds)
                                # '2' = Open
                                # '3' = High
                                # '4' = Low
                                # '5' = Close
                                # '6' = Volume (actual shares)
                                # '7' = Time (ms since epoch)
                                # '8' = Day

                                symbol = content.get("key")
                                open_price = content.get("2")
                                high_price = content.get("3")
                                low_price = content.get("4")
                                close_price = content.get("5")
                                volume = content.get("6")  # Actual volume in shares
                                timestamp = content.get("7")  # Bar time from server

                                # Convert timestamps to readable format with milliseconds
                                from datetime import datetime

                                # Bar time (from server)
                                if timestamp:
                                    dt = datetime.fromtimestamp(timestamp / 1000)
                                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Include milliseconds
                                else:
                                    time_str = "N/A"

                                # Reception time (when we received it)
                                dt_reception = datetime.fromtimestamp(reception_time)
                                reception_str = dt_reception.strftime("%Y-%m-%d %H:%M:%S.%f")[
                                    :-3
                                ]  # Include milliseconds

                                print(f"Symbol: {symbol}")
                                print(f"  Bar Time:       {time_str}")
                                print(f"  Reception Time: {reception_str}")
                                print(f"  Open:   ${open_price:.2f}" if open_price else "  Open:   N/A")
                                print(f"  High:   ${high_price:.2f}" if high_price else "  High:   N/A")
                                print(f"  Low:    ${low_price:.2f}" if low_price else "  Low:    N/A")
                                print(f"  Close:  ${close_price:.2f}" if close_price else "  Close:  N/A")
                                print(f"  Volume: {volume:,}" if volume else "  Volume: N/A")
                elif "notify" in data:
                    # Heartbeat messages (suppress or show based on preference)
                    pass  # Uncomment next line to see heartbeats
                    # print(f"Heartbeat: {data['notify'][0]['heartbeat']}")
                else:
                    print(f"Other message: {json.dumps(data, indent=2)}")

        except KeyboardInterrupt:
            print("\n\nShutting down...")

            # Step 4: LOGOUT
            logout_request = {
                "requests": [
                    {
                        "requestid": "3",
                        "service": "ADMIN",
                        "command": "LOGOUT",
                        "SchwabClientCustomerId": customer_id,
                        "SchwabClientCorrelId": correl_id,
                        "parameters": {},
                    }
                ]
            }

            await websocket.send(json.dumps(logout_request))
            response = await websocket.recv()
            # Decode bytes if needed, or use repr for debugging
            response_str = response.decode() if isinstance(response, bytes) else str(response)
            print(f"LOGOUT response: {response_str}")


async def main() -> None:
    """Main entry point."""
    # Check for --clear-cache flag
    import sys

    if "--clear-cache" in sys.argv:
        if TOKEN_CACHE_FILE.exists():
            TOKEN_CACHE_FILE.unlink()
            print("✓ Token cache cleared\n")
        else:
            print("⚠ No token cache found\n")
        return

    print("=" * 70)
    print("Schwab API Demo - AAPL Data")
    print("=" * 70)
    print()
    print("Choose mode:")
    print("  1. Historical data (REST API - daily/minute candles)")
    print("  2. Streaming data (WebSocket - live 1-minute candles)")
    print("  3. Check data availability (how far back does history go?)")
    print()

    # Get mode from user or environment variable
    mode = os.environ.get("SCHWAB_MODE", "").lower()
    if mode not in ["historical", "streaming", "availability", "1", "2", "3"]:
        mode_input = input("Enter mode [1/2/3]: ").strip()
        if mode_input == "1":
            mode = "historical"
        elif mode_input == "2":
            mode = "streaming"
        elif mode_input == "3":
            mode = "availability"
        else:
            mode = "historical"
    elif mode in ["1", "historical"]:
        mode = "historical"
    elif mode in ["3", "availability"]:
        mode = "availability"
    else:
        mode = "streaming"

    print(f"\n{'=' * 70}")
    print(f"Mode: {mode.upper()}")
    print("=" * 70)

    # Step 1: Authenticate and get access token
    access_token = get_access_token()

    if mode == "availability":
        # Data availability check mode
        print("\n" + "=" * 70)
        print("Data Availability Check")
        print("=" * 70)

        symbol = input("\nSymbol (default: AAPL): ").strip().upper() or "AAPL"

        print("\nFrequency Type:")
        print("  1. minute (max 10 days)")
        print("  2. daily (max 20 years) - default")
        print("  3. weekly (max 20 years)")
        print("  4. monthly (max 20 years)")
        freq_type_choice = input("Choice [1-4]: ").strip()
        freq_type_map = {"1": "minute", "2": "daily", "3": "weekly", "4": "monthly"}
        frequency_type = freq_type_map.get(freq_type_choice, "daily")

        check_data_availability(access_token, symbol, frequency_type)

    elif mode == "historical":
        # Historical data mode - REST API
        print("\n" + "=" * 70)
        print("Historical Data Options")
        print("=" * 70)

        # Get parameters
        symbol = input("\nSymbol (default: AAPL): ").strip().upper() or "AAPL"

        print("\nPeriod Type:")
        print("  1. day")
        print("  2. month (default)")
        print("  3. year")
        print("  4. ytd (year-to-date)")
        period_type_choice = input("Choice [1-4]: ").strip()
        period_type_map = {"1": "day", "2": "month", "3": "year", "4": "ytd"}
        period_type = period_type_map.get(period_type_choice, "month")

        period_input = input(f"Number of {period_type}s (default: 1): ").strip()
        period: int = int(period_input) if period_input else 1

        print("\nFrequency Type:")
        print("  1. minute")
        print("  2. daily (default)")
        print("  3. weekly")
        print("  4. monthly")
        freq_type_choice = input("Choice [1-4]: ").strip()
        freq_type_map = {"1": "minute", "2": "daily", "3": "weekly", "4": "monthly"}
        frequency_type = freq_type_map.get(freq_type_choice, "daily")

        if frequency_type == "minute":
            print("\nFrequency (minutes):")
            print("  1, 5, 10, 15, 30")
            frequency_input = input("Choice (default: 5): ").strip()
            frequency: int = int(frequency_input) if frequency_input else 5
        else:
            frequency = 1

        print()

        # Fetch historical data
        data = get_historical_data(access_token, symbol, period_type, period, frequency_type, frequency)

        # Display results
        limit_input = input("Show how many recent candles? (default: 10, 'all' for all): ").strip()
        if limit_input.lower() == "all":
            limit = None
        else:
            limit = int(limit_input) if limit_input else 10

        display_historical_candles(data, limit)

    else:
        # Streaming mode - WebSocket
        # Step 2: Get user preferences with streamer info
        prefs = get_user_preferences(access_token)

        # Extract streamer connection details
        streamer_info = prefs["streamerInfo"][0]
        streamer_url = streamer_info["streamerSocketUrl"]
        customer_id = streamer_info["schwabClientCustomerId"]
        correl_id = streamer_info["schwabClientCorrelId"]

        print(f"Streamer URL: {streamer_url}")
        print(f"Customer ID: {customer_id}")
        print(f"Correlation ID: {correl_id}\n")

        # Step 3: Connect and stream data
        await stream_chart_data(access_token, streamer_url, customer_id, correl_id)


if __name__ == "__main__":
    asyncio.run(main())
