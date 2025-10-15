#!/usr/bin/env python3
"""
Test Schwab OAuth authentication (manual mode).

This script tests the OAuth flow with manual code entry,
which is useful when the callback URI at Schwab doesn't match
the local development environment.

Usage:
    python scripts/test_schwab_oauth.py

Environment Variables:
    SCHWAB_API_KEY: Your Schwab API key (client_id)
    SCHWAB_API_SECRET: Your Schwab API secret

Example:
    export SCHWAB_API_KEY="your_key_here"
    export SCHWAB_API_SECRET="your_secret_here"
    python scripts/test_schwab_oauth.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrader.auth.schwab_oauth import SchwabOAuthManager


def main():
    """Test Schwab OAuth with manual code entry."""
    # Get credentials from environment
    client_id = os.getenv("SCHWAB_API_KEY")
    client_secret = os.getenv("SCHWAB_API_SECRET")

    if not client_id or not client_secret:
        print("❌ Error: Missing environment variables")
        print("\nRequired:")
        print("  - SCHWAB_API_KEY")
        print("  - SCHWAB_API_SECRET")
        print("\nSet them with:")
        print('  export SCHWAB_API_KEY="your_key"')
        print('  export SCHWAB_API_SECRET="your_secret"')
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SCHWAB OAUTH TEST (MANUAL MODE)")
    print("=" * 70)
    print("\nThis will test OAuth authentication with manual code entry.")
    print("You'll need to copy the authorization code from your browser.\n")

    # Get the registered callback URI
    print("Enter your registered callback URI at Schwab")
    print("(e.g., https://analytic-alpha.com)")
    redirect_uri = input("\n> ").strip()

    if not redirect_uri:
        print("❌ Callback URI required")
        sys.exit(1)

    # Create OAuth manager in manual mode
    manager = SchwabOAuthManager(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        manual_mode=True,  # Enable manual code entry
    )

    try:
        # Try to get token from cache first
        token = manager.get_access_token()

        print("\n" + "=" * 70)
        print("✅ SUCCESS! OAuth authentication complete")
        print("=" * 70)
        print(f"\nAccess Token: {token[:20]}...{token[-20:]}")
        print(f"Token Length: {len(token)} characters")
        print(f"\nToken cached at: {manager.token_cache_path}")
        print("\n💡 Next steps:")
        print("  1. Token is cached - future calls will reuse it")
        print("  2. Token expires after 30 minutes")
        print("  3. Refresh token lasts 7 days")
        print("=" * 70)

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR: OAuth authentication failed")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\n🔍 Troubleshooting:")
        print("  1. Check your API credentials")
        print("  2. Verify the authorization code was copied correctly")
        print("  3. Make sure the callback URI matches Schwab configuration")
        print("  4. Check if you authorized the application")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
