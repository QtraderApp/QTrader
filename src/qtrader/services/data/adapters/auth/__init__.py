"""
Authentication modules for DataService adapters.

This package handles OAuth authentication and SSL certificate management
for data providers.

Modules:
    ssl_generator: Generate self-signed SSL certificates for localhost
"""

from qtrader.services.data.adapters.auth.ssl_generator import ensure_ssl_certificates, generate_self_signed_cert

__all__ = [
    "ensure_ssl_certificates",
    "generate_self_signed_cert",
]
