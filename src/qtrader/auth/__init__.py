"""
Authentication modules for QTrader.

This package handles OAuth authentication and SSL certificate management
for various data providers.

Modules:
    ssl_generator: Generate self-signed SSL certificates for localhost
    schwab_oauth: Schwab OAuth 2.0 authentication with HTTPS callback
"""

from qtrader.auth.schwab_oauth import SchwabOAuthManager
from qtrader.auth.ssl_generator import ensure_ssl_certificates, generate_self_signed_cert

__all__ = [
    "SchwabOAuthManager",
    "ensure_ssl_certificates",
    "generate_self_signed_cert",
]
