# Schwab Integration Plan

**Status:** Planning → Implementation\
**Date:** October 15, 2025\
**Branch:** `feature/schwab-integration`

______________________________________________________________________

## ✅ Decisions Approved

1. **MultiBar Strategy:** Option 1 - Partial MultiBar (None for unavailable modes) ✓
1. **OAuth Callback:** **HTTPS required** (127.0.0.1:8182 with self-signed cert) ✓
1. **Token Storage:** Plain JSON in `~/.qtrader/schwab_tokens.json` (chmod 600) ✓
1. **Rate Limiting:** 10 requests/second - implement exponential backoff ✓
1. **Data Validation:** No cross-validation with Algoseek ✓
1. **Error Handling:** Hard fail with descriptive error showing available date ranges ✓
1. **Symbol Mapping:** Handle later (not now) ✓
1. **Metadata:** JSON format with best practices ✓
1. **Update Strategy:** On-demand per ticker + `--update-all` flag ✓

______________________________________________________________________

## 🏗️ Architecture

### Component Overview

```
schwab/
├── auth/
│   ├── oauth_manager.py       # HTTPS callback + token management
│   └── ssl_cert_generator.py  # Self-signed cert for localhost
├── adapters/
│   └── schwab.py              # Cache-first adapter with API client
├── models/vendors/
│   └── schwab.py              # SchwabBar + SchwabPriceSeries
└── cache/
    └── metadata_manager.py    # Cache metadata tracking
```

### Data Flow

```
CLI Request
    ↓
DataSourceResolver (config/data_sources.yaml)
    ↓
SchwabOHLCAdapter
    ↓
Check Local Cache (data/us-equity-daily-adjusted-schwab/SYMBOL/)
    ├─ Cache Hit (all dates) → Return from cache
    ├─ Partial Hit → API fetch missing dates → Merge & cache
    └─ Cache Miss → API fetch all dates → Cache
    ↓
API Fetch (if needed)
    ├─ OAuth (via HTTPS callback)
    ├─ Rate limiting (10/sec)
    ├─ Exponential backoff on errors
    └─ Data validation
    ↓
SchwabBar (vendor model)
    ↓
SchwabPriceSeries.to_canonical()
    ↓
(None, PriceSeries, None)  # Only adjusted mode
    ↓
MultiBar (unadjusted=None, adjusted=Bar, total_return=None)
    ↓
CLI Display
```

______________________________________________________________________

## 📁 File Structure

```
src/qtrader/
├── auth/
│   ├── __init__.py
│   ├── schwab_oauth.py              # NEW: OAuth manager with HTTPS
│   └── ssl_generator.py             # NEW: Self-signed cert generator
├── adapters/
│   ├── __init__.py
│   ├── algoseek.py                  # Existing
│   ├── schwab.py                    # NEW: Schwab adapter
│   └── resolver.py                  # UPDATE: Register schwab
├── models/vendors/
│   ├── __init__.py
│   ├── algoseek.py                  # Existing
│   └── schwab.py                    # NEW: Schwab models
└── cli.py                           # UPDATE: Handle None modes

config/
├── data_sources.yaml                # UPDATE: Add schwab config
└── .envrc.example                   # UPDATE: Add Schwab env vars

data/
└── us-equity-daily-adjusted-schwab/ # NEW: Cache directory
    ├── AAPL/
    │   ├── data.parquet
    │   └── .metadata.json           # Last update, date range, row count
    └── MSFT/
        ├── data.parquet
        └── .metadata.json

.qtrader/                            # User config directory
├── ssl/
│   ├── localhost.pem                # Self-signed certificate
│   └── localhost-key.pem            # Private key
└── schwab_tokens.json               # OAuth tokens (chmod 600)
```

______________________________________________________________________

## 🔐 OAuth Implementation (HTTPS)

### SSL Certificate Generation

```python
# src/qtrader/auth/ssl_generator.py
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
from pathlib import Path

def generate_self_signed_cert(output_dir: Path) -> tuple[Path, Path]:
    """Generate self-signed SSL certificate for localhost."""

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Create certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QTrader"),
        x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Save files
    cert_path = output_dir / "localhost.pem"
    key_path = output_dir / "localhost-key.pem"

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Write private key
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Set restrictive permissions
    cert_path.chmod(0o644)
    key_path.chmod(0o600)

    return cert_path, key_path
```

### OAuth Manager with HTTPS Callback

```python
# src/qtrader/auth/schwab_oauth.py
import ssl
import json
import time
import base64
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import urllib.parse
import requests

class SchwabOAuthManager:
    """Manage Schwab OAuth tokens with HTTPS callback server."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "https://127.0.0.1:8182",
        token_cache_path: Path | None = None,
        ssl_cert_dir: Path | None = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

        # Token cache
        self.token_cache_path = token_cache_path or (
            Path.home() / ".qtrader" / "schwab_tokens.json"
        )

        # SSL certificates
        self.ssl_cert_dir = ssl_cert_dir or (
            Path.home() / ".qtrader" / "ssl"
        )

        # Ensure directories exist
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.ssl_cert_dir.mkdir(parents=True, exist_ok=True)

        # OAuth endpoints
        self.auth_url = "https://api.schwabapi.com/v1/oauth/authorize"
        self.token_url = "https://api.schwabapi.com/v1/oauth/token"

    def get_access_token(self, force_new: bool = False) -> str:
        """Get valid access token (from cache or new OAuth flow)."""

        if not force_new:
            cached_token = self._load_cached_token()
            if cached_token:
                return cached_token

        # No valid cache - start OAuth flow
        return self._oauth_flow()

    def _oauth_flow(self) -> str:
        """Execute OAuth flow with HTTPS callback."""

        # Step 1: Generate authorization URL
        auth_params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
        }
        auth_url = f"{self.auth_url}?{urllib.parse.urlencode(auth_params)}"

        print("\n" + "=" * 70)
        print("SCHWAB OAUTH AUTHENTICATION")
        print("=" * 70)
        print(f"\nPlease visit this URL to authorize:\n\n{auth_url}\n")
        print("A local HTTPS server will capture the authorization code...")
        print("(You may see a browser security warning - this is expected)\n")

        # Step 2: Start HTTPS callback server
        auth_code = self._start_callback_server()

        if not auth_code:
            raise RuntimeError("Failed to capture authorization code")

        # Step 3: Exchange code for token
        return self._exchange_code_for_token(auth_code)

    def _start_callback_server(self) -> str | None:
        """Start HTTPS server to capture OAuth callback."""

        # Ensure SSL certificates exist
        cert_path, key_path = self._ensure_ssl_certs()

        # Create SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_path, key_path)

        # Callback handler
        auth_code = {"code": None}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)

                if "code" in params:
                    auth_code["code"] = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<h1>Authorization successful!</h1>")
                    self.wfile.write(b"<p>You can close this window.</p>")
                else:
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logs

        # Start server
        server = HTTPServer(("127.0.0.1", 8182), CallbackHandler)
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

        # Handle request in thread
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # Wait for callback (2 minute timeout)
        server_thread.join(timeout=120)

        return auth_code["code"]

    def _exchange_code_for_token(self, auth_code: str) -> str:
        """Exchange authorization code for access token."""

        credentials = f"{self.client_id}:{self.client_secret}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {b64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
        }

        response = requests.post(self.token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()

        # Cache token
        self._save_token_cache(token_data)

        return token_data["access_token"]

    def _ensure_ssl_certs(self) -> tuple[Path, Path]:
        """Ensure SSL certificates exist, generate if needed."""

        cert_path = self.ssl_cert_dir / "localhost.pem"
        key_path = self.ssl_cert_dir / "localhost-key.pem"

        if not cert_path.exists() or not key_path.exists():
            from qtrader.auth.ssl_generator import generate_self_signed_cert
            print("Generating self-signed SSL certificate...")
            cert_path, key_path = generate_self_signed_cert(self.ssl_cert_dir)
            print(f"✓ Certificate created: {cert_path}")

        return cert_path, key_path

    def _load_cached_token(self) -> str | None:
        """Load token from cache if valid."""

        if not self.token_cache_path.exists():
            return None

        try:
            with open(self.token_cache_path) as f:
                token_data = json.load(f)

            expires_at = token_data.get("expires_at", 0)
            if time.time() < expires_at:
                return token_data["access_token"]
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    def _save_token_cache(self, token_data: dict) -> None:
        """Save token to cache file."""

        expires_in = token_data.get("expires_in", 1800)
        cache_data = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": expires_in,
            "expires_at": time.time() + expires_in,
            "token_type": token_data.get("token_type", "Bearer"),
            "created_at": time.time(),
        }

        with open(self.token_cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        self.token_cache_path.chmod(0o600)
```

______________________________________________________________________

## 📊 Metadata Format

```json
{
  "symbol": "AAPL",
  "source": "schwab",
  "last_updated": "2025-10-15T10:30:45Z",
  "date_range": {
    "start": "2019-01-01",
    "end": "2025-10-15"
  },
  "row_count": 1685,
  "data_file": "data.parquet",
  "data_file_size_bytes": 125840,
  "checksum": "sha256:abc123...",
  "api_version": "v1",
  "fetch_history": [
    {
      "timestamp": "2025-10-15T10:30:45Z",
      "date_range": {
        "start": "2019-01-01",
        "end": "2025-10-15"
      },
      "rows_fetched": 1685,
      "api_calls": 2
    }
  ],
  "data_quality": {
    "gaps": [],
    "last_validation": "2025-10-15T10:30:45Z"
  }
}
```

______________________________________________________________________

## 🚦 Rate Limiting Strategy

### Implementation (10 requests/second)

```python
import time
from collections import deque

class RateLimiter:
    """Token bucket rate limiter - 10 requests/second."""

    def __init__(self, rate: int = 10, period: float = 1.0):
        self.rate = rate
        self.period = period
        self.timestamps = deque(maxlen=rate)

    def acquire(self) -> None:
        """Block until request can proceed."""

        now = time.time()

        # Remove old timestamps outside window
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()

        # If at capacity, wait
        if len(self.timestamps) >= self.rate:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.timestamps.popleft()

        self.timestamps.append(time.time())

class ExponentialBackoff:
    """Exponential backoff for retries."""

    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempts = 0

    def wait(self) -> None:
        """Wait with exponential backoff."""
        delay = min(self.base_delay * (2 ** self.attempts), self.max_delay)
        time.sleep(delay)
        self.attempts += 1

    def reset(self) -> None:
        """Reset attempt counter."""
        self.attempts = 0
```

______________________________________________________________________

## 🛠️ Implementation Phases

### Phase 1: Foundation (2-3 hours)

- [x] Planning document created
- [ ] Create directory structure
- [ ] Implement SSL certificate generator
- [ ] Implement OAuth manager with HTTPS
- [ ] Add environment variable support
- [ ] Write unit tests for OAuth

### Phase 2: Vendor Models (2-3 hours)

- [ ] Create `SchwabBar` model
- [ ] Create `SchwabPriceSeries` with adjusted-only logic
- [ ] Implement validation
- [ ] Write unit tests

### Phase 3: Adapter Core (4-5 hours)

- [ ] Create `SchwabOHLCAdapter` skeleton
- [ ] Implement cache reading logic
- [ ] Implement API client
- [ ] Implement rate limiting
- [ ] Implement exponential backoff
- [ ] Write unit tests

### Phase 4: Caching Logic (3-4 hours)

- [ ] Implement metadata manager
- [ ] Implement incremental fetch logic
- [ ] Implement cache writing
- [ ] Implement cache validation
- [ ] Write integration tests

### Phase 5: Integration (2-3 hours)

- [ ] Update `config/data_sources.yaml`
- [ ] Update `DataSourceResolver`
- [ ] Update CLI for None mode handling
- [ ] Add `--update-all` flag
- [ ] Write end-to-end tests

### Phase 6: Polish (2-3 hours)

- [ ] Error messages with date ranges
- [ ] Logging enhancements
- [ ] Documentation
- [ ] Example usage
- [ ] Performance testing

**Total Estimated Time:** 15-21 hours (~2-3 days)

______________________________________________________________________

## 🧪 Testing Strategy

### Unit Tests

- OAuth token management
- SSL certificate generation
- Rate limiting
- Exponential backoff
- Vendor models validation
- Metadata management

### Integration Tests

- Cache hit/miss scenarios
- Incremental fetch logic
- API error handling
- Token refresh flow

### End-to-End Tests

- Full CLI workflow
- Multi-symbol caching
- Update strategies

______________________________________________________________________

## 📝 Configuration

```yaml
# config/data_sources.yaml
data_sources:
  schwab:
    adapter: schwabOHLC
    cache_root: "data/us-equity-daily-adjusted-schwab"
    mode: adjusted_only

    api:
      base_url: "https://api.schwabapi.com"
      auth:
        client_id: "${SCHWAB_API_KEY}"
        client_secret: "${SCHWAB_API_SECRET}"
        redirect_uri: "https://127.0.0.1:8182"
        token_cache: "~/.qtrader/schwab_tokens.json"

      rate_limit:
        requests_per_second: 10
        backoff_base_delay: 1.0
        backoff_max_delay: 60.0

      fetch:
        chunk_size_days: 365
        max_years: 20
        timeout_seconds: 30
```

```bash
# .envrc.example
export SCHWAB_API_KEY="your_api_key_here"
export SCHWAB_API_SECRET="your_api_secret_here"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"
```

______________________________________________________________________

## 🎯 Success Criteria

1. ✅ HTTPS callback server works for OAuth
1. ✅ Tokens cached and auto-refreshed
1. ✅ Cache-first logic prevents redundant API calls
1. ✅ Rate limiting enforced (10 req/sec)
1. ✅ Incremental fetch only missing dates
1. ✅ Hard fail with descriptive errors when API needed
1. ✅ CLI handles None modes gracefully
1. ✅ `--update-all` flag works
1. ✅ All tests pass
1. ✅ Documentation complete

______________________________________________________________________

## 🚀 Next Steps

Ready to start implementation! Shall I begin with Phase 1 (OAuth foundation)?

**Commands to run:**

```bash
# Create feature branch
git checkout -b feature/schwab-integration

# Install dependencies
pip install cryptography  # For SSL certificate generation

# Start implementation
# Phase 1: OAuth + SSL
```

Proceed with implementation? 🎯
