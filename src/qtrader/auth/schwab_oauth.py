"""
Schwab OAuth manager with HTTPS callback server.

This module handles OAuth 2.0 authentication for Schwab API:
- HTTPS callback server (required by Schwab)
- Token acquisition and caching
- Automatic token refresh
- Secure token storage

The OAuth flow:
1. Generate authorization URL
2. User opens URL in browser and authorizes
3. Schwab redirects to https://127.0.0.1:8182 with authorization code
4. Local HTTPS server captures the code
5. Exchange code for access token
6. Cache token for reuse
"""

import base64
import json
import ssl
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import requests

from qtrader.auth.ssl_generator import ensure_ssl_certificates
from qtrader.config.logging_config import LoggerFactory

logger = LoggerFactory.get_logger()


class SchwabOAuthManager:
    """
    Manage Schwab OAuth tokens with HTTPS callback server.

    This class handles the complete OAuth flow including:
    - HTTPS callback server with self-signed certificates
    - Token acquisition from authorization code
    - Token caching in user's home directory
    - Automatic token refresh before expiry

    Configuration:
        - Requires SCHWAB_API_KEY (client_id)
        - Requires SCHWAB_API_SECRET (client_secret)
        - Redirect URI: https://127.0.0.1:8182 (must match Schwab config)
        - Token cache: ~/.qtrader/schwab_tokens.json (chmod 600)
        - SSL certs: ~/.qtrader/ssl/localhost.pem (auto-generated)

    Security:
        - Tokens stored with 600 permissions (owner read/write only)
        - SSL certificates auto-generated for localhost
        - Browser will show security warning (expected for self-signed)

    Example:
        >>> manager = SchwabOAuthManager(
        ...     client_id=os.getenv("SCHWAB_API_KEY"),
        ...     client_secret=os.getenv("SCHWAB_API_SECRET"),
        ... )
        >>> token = manager.get_access_token()
        >>> # Use token for API requests
        >>> headers = {"Authorization": f"Bearer {token}"}
    """

    # OAuth endpoints
    AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
    TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

    # Callback server config
    CALLBACK_HOST = "127.0.0.1"
    CALLBACK_PORT = 8182

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str | None = None,
        token_cache_path: Path | None = None,
        ssl_cert_dir: Path | None = None,
        manual_mode: bool = False,
    ):
        """
        Initialize Schwab OAuth manager.

        Args:
            client_id: Schwab API key (from developer portal)
            client_secret: Schwab API secret
            redirect_uri: OAuth redirect URI (default: https://127.0.0.1:8182)
            token_cache_path: Path to token cache file (default: ~/.qtrader/schwab_tokens.json)
            ssl_cert_dir: Directory for SSL certificates (default: ~/.qtrader/ssl)
            manual_mode: If True, use manual code entry instead of callback server

        Raises:
            ValueError: If client_id or client_secret is empty
        """
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret are required")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri or f"https://{self.CALLBACK_HOST}:{self.CALLBACK_PORT}"
        self.manual_mode = manual_mode

        # Token cache path
        if token_cache_path is None:
            token_cache_path = Path.home() / ".qtrader" / "schwab_tokens.json"
        self.token_cache_path = token_cache_path

        # SSL certificate directory
        if ssl_cert_dir is None:
            ssl_cert_dir = Path.home() / ".qtrader" / "ssl"
        self.ssl_cert_dir = ssl_cert_dir

        # Ensure cache directory exists
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "schwab_oauth.manager_initialized",
            redirect_uri=self.redirect_uri,
            token_cache=str(self.token_cache_path),
        )

    def get_access_token(self, force_new: bool = False) -> str:
        """
        Get valid access token (from cache or new OAuth flow).

        Checks cache first. If token is expired or force_new=True,
        initiates new OAuth flow.

        Args:
            force_new: Force new authentication even if cached token exists

        Returns:
            Valid access token string

        Raises:
            RuntimeError: If OAuth flow fails
            requests.HTTPError: If token exchange fails

        Example:
            >>> token = manager.get_access_token()
            >>> # Token is valid for ~30 minutes
            >>> # Subsequent calls return cached token
            >>> token2 = manager.get_access_token()  # Returns cached
        """
        if not force_new:
            cached_token = self._load_cached_token()
            if cached_token:
                logger.info("schwab_oauth.using_cached_token")
                return cached_token

        logger.info("schwab_oauth.starting_oauth_flow")
        return self._oauth_flow()

    def _oauth_flow(self) -> str:
        """
        Execute complete OAuth flow with HTTPS callback.

        Steps:
        1. Generate authorization URL
        2. Start HTTPS callback server
        3. User authorizes in browser
        4. Capture authorization code
        5. Exchange code for token
        6. Cache token

        Returns:
            Access token string

        Raises:
            RuntimeError: If callback server fails or times out
        """
        # Step 1: Generate authorization URL
        auth_params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
        }
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

        print("\n" + "=" * 70)
        print("SCHWAB OAUTH AUTHENTICATION REQUIRED")
        print("=" * 70)
        print(f"\n📋 Authorization URL:\n\n{auth_url}\n")
        print("🔐 Steps:")
        print("  1. Click the URL above (or copy to browser)")
        print("  2. Log in to your Schwab account")
        print("  3. Authorize the application")

        if self.manual_mode:
            print("  4. Copy the 'code' parameter from the redirect URL")
            print("     (URL will look like: https://...?code=XXXXX)")
            print("\n⏳ Waiting for code input...")
        else:
            print("  4. You'll be redirected to a local page")
            print(f"     (https://{self.CALLBACK_HOST}:{self.CALLBACK_PORT})")
            print("\n⚠️  Browser Security Warning:")
            print("  - You will see a security warning (expected)")
            print("  - This is because we use a self-signed certificate")
            print("  - Click 'Advanced' → 'Proceed to 127.0.0.1'")
            print("\n⏳ Waiting for authorization...")
        print("=" * 70)

        # Step 2: Get authorization code (manual or callback)
        if self.manual_mode:
            auth_code = self._get_code_manual()
        else:
            auth_code = self._start_callback_server()

        if not auth_code:
            raise RuntimeError("Failed to capture authorization code from callback")

        print("\n✅ Authorization code received!")

        # Step 3: Exchange code for token
        return self._exchange_code_for_token(auth_code)

    def _get_code_manual(self) -> str | None:
        """
        Get authorization code via manual user input.

        This method prompts the user to paste the authorization code
        from the redirect URL. Useful when callback server cannot be used.

        Returns:
            Authorization code from user input, or None if cancelled

        Note:
            User should copy the 'code' parameter from the URL after authorization
        """
        print("\n📝 After authorizing, paste the full redirect URL here:")
        print("   (or just the code parameter)")

        try:
            user_input = input("\n> ").strip()

            if not user_input:
                logger.warning("schwab_oauth.manual_code_empty")
                return None

            # Try to parse as URL first
            if "code=" in user_input:
                parsed = urllib.parse.urlparse(user_input)
                params = urllib.parse.parse_qs(parsed.query)
                if "code" in params:
                    code = params["code"][0]
                    logger.info("schwab_oauth.manual_code_extracted", code_length=len(code))
                    return code

            # Assume it's just the code
            logger.info("schwab_oauth.manual_code_received", code_length=len(user_input))
            return user_input

        except (KeyboardInterrupt, EOFError):
            logger.warning("schwab_oauth.manual_code_cancelled")
            return None

    def _start_callback_server(self) -> str | None:
        """
        Start HTTPS server to capture OAuth callback.

        Starts a temporary HTTPS server that listens for Schwab's
        OAuth redirect with the authorization code.

        Returns:
            Authorization code from callback, or None if timeout

        Note:
            Server runs for max 120 seconds (2 minutes)
        """
        logger.info(
            "schwab_oauth.starting_callback_server",
            host=self.CALLBACK_HOST,
            port=self.CALLBACK_PORT,
        )

        # Ensure SSL certificates exist
        cert_path, key_path = ensure_ssl_certificates(self.ssl_cert_dir)

        # Create SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_path, key_path)

        # Container to capture auth code from handler
        auth_code_container: dict[str, str | None] = {"code": None}

        # Define callback handler
        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle OAuth callback GET request."""
                # Parse query parameters
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)

                if "code" in params:
                    # Capture authorization code
                    auth_code_container["code"] = params["code"][0]

                    # Send success response
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

                    html = """
                    <html>
                    <head><title>QTrader - Authorization Successful</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                        <h1 style="color: green;">✅ Authorization Successful!</h1>
                        <p>You can close this window and return to the terminal.</p>
                        <p style="color: #666; font-size: 12px;">QTrader OAuth Callback</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())

                    logger.info("schwab_oauth.authorization_code_captured")
                else:
                    # Error in callback
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

                    html = """
                    <html>
                    <head><title>QTrader - Authorization Failed</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                        <h1 style="color: red;">❌ Authorization Failed</h1>
                        <p>No authorization code received.</p>
                        <p>Please try again.</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())

                    logger.warning("schwab_oauth.callback_missing_code", params=params)

            def log_message(self, format, *args):
                """Suppress default HTTP logging."""
                pass

        # Create and configure server
        server = HTTPServer((self.CALLBACK_HOST, self.CALLBACK_PORT), CallbackHandler)
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

        # Run server in thread with timeout
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # Wait for callback (2 minute timeout)
        server_thread.join(timeout=120)

        if not auth_code_container["code"]:
            logger.error("schwab_oauth.callback_timeout")

        return auth_code_container["code"]

    def _exchange_code_for_token(self, auth_code: str) -> str:
        """
        Exchange authorization code for access token.

        Args:
            auth_code: Authorization code from OAuth callback

        Returns:
            Access token string

        Raises:
            requests.HTTPError: If token exchange fails
        """
        logger.info("schwab_oauth.exchanging_code_for_token")

        # Prepare credentials
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

        try:
            response = requests.post(self.TOKEN_URL, headers=headers, data=data, timeout=30)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(
                "schwab_oauth.token_exchange_failed",
                status_code=e.response.status_code if e.response else None,
                error=str(e),
            )
            print("\n❌ Token exchange failed!")
            if e.response:
                print(f"Status: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            print("\nCommon issues:")
            print("  1. Authorization code already used (single-use only)")
            print("  2. Redirect URI mismatch")
            print("  3. Code expired (codes expire quickly)")
            print("  4. Invalid API credentials")
            raise

        token_data = response.json()
        access_token: str = token_data["access_token"]

        # Cache token for future use
        self._save_token_cache(token_data)

        logger.info(
            "schwab_oauth.token_obtained",
            expires_in=token_data.get("expires_in", "unknown"),
        )
        print(f"\n✅ Access token obtained (expires in {token_data.get('expires_in', '?')}s)")
        print(f"💾 Token cached: {self.token_cache_path}\n")

        return access_token

    def _load_cached_token(self) -> str | None:
        """
        Load token from cache if valid.

        Returns:
            Access token if valid, None if expired or missing
        """
        if not self.token_cache_path.exists():
            logger.debug("schwab_oauth.no_cached_token")
            return None

        try:
            with open(self.token_cache_path) as f:
                token_data = json.load(f)

            expires_at = token_data.get("expires_at", 0)
            now = time.time()

            if now < expires_at:
                # Token still valid
                remaining = int(expires_at - now)
                logger.info(
                    "schwab_oauth.cached_token_valid",
                    remaining_seconds=remaining,
                )
                access_token: str = token_data["access_token"]
                return access_token
            else:
                logger.info("schwab_oauth.cached_token_expired")
                return None

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("schwab_oauth.cache_read_error", error=str(e))
            return None

    def _save_token_cache(self, token_data: dict) -> None:
        """
        Save token to cache file with secure permissions.

        Args:
            token_data: Token response from Schwab API
        """
        expires_in = token_data.get("expires_in", 1800)  # Default 30 min
        cache_data = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": expires_in,
            "expires_at": time.time() + expires_in,
            "token_type": token_data.get("token_type", "Bearer"),
            "created_at": time.time(),
            "scope": token_data.get("scope", ""),
        }

        # Write to temporary file first
        temp_path = self.token_cache_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Set restrictive permissions (owner only)
        temp_path.chmod(0o600)

        # Atomic rename
        temp_path.replace(self.token_cache_path)

        logger.info(
            "schwab_oauth.token_cached",
            path=str(self.token_cache_path),
            expires_in_minutes=expires_in // 60,
        )
