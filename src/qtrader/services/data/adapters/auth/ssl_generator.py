"""
SSL certificate generator for local HTTPS OAuth callback server.

This module generates self-signed SSL certificates for localhost to enable
HTTPS callback URLs required by Schwab OAuth flow.

The certificates are stored in ~/.qtrader/ssl/ and reused across sessions.
Browser security warnings are expected for self-signed certificates.
"""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from qtrader.system import LoggerFactory

logger = LoggerFactory.get_logger()


def generate_self_signed_cert(output_dir: Path) -> tuple[Path, Path]:
    """
    Generate self-signed SSL certificate for localhost.

    Creates a certificate valid for 1 year with:
    - Common Name: 127.0.0.1
    - Subject Alternative Names: localhost, 127.0.0.1
    - 2048-bit RSA key

    Args:
        output_dir: Directory to store certificate files

    Returns:
        Tuple of (cert_path, key_path)

    Example:
        >>> cert_dir = Path.home() / ".qtrader" / "ssl"
        >>> cert_path, key_path = generate_self_signed_cert(cert_dir)
        >>> print(f"Certificate: {cert_path}")
        >>> print(f"Private key: {key_path}")

    Note:
        Browsers will show security warnings for self-signed certificates.
        This is expected and can be safely ignored for localhost development.
    """
    logger.info("ssl_generator.generating_certificate", output_dir=str(output_dir))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate private key
    logger.debug("ssl_generator.generating_private_key")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Create certificate subject/issuer (same for self-signed)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QTrader"),
            x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1"),
        ]
    )

    # Build certificate
    logger.debug("ssl_generator.building_certificate")
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Define output paths
    cert_path = output_dir / "localhost.pem"
    key_path = output_dir / "localhost-key.pem"

    # Write certificate
    logger.debug("ssl_generator.writing_certificate", path=str(cert_path))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Write private key
    logger.debug("ssl_generator.writing_private_key", path=str(key_path))
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Set restrictive permissions
    cert_path.chmod(0o644)  # Read for all, write for owner
    key_path.chmod(0o600)  # Read/write for owner only

    logger.info(
        "ssl_generator.certificate_generated",
        cert_path=str(cert_path),
        key_path=str(key_path),
        valid_days=365,
    )

    return cert_path, key_path


def ensure_ssl_certificates(cert_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Ensure SSL certificates exist, generate if needed.

    Checks if valid certificates exist in the specified directory.
    If not found or expired, generates new ones.

    Args:
        cert_dir: Directory for certificates (default: ~/.qtrader/ssl)

    Returns:
        Tuple of (cert_path, key_path)

    Example:
        >>> cert_path, key_path = ensure_ssl_certificates()
        >>> # Use certificates for HTTPS server
        >>> import ssl
        >>> ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        >>> ssl_context.load_cert_chain(cert_path, key_path)

    Note:
        This function is idempotent - safe to call multiple times.
        Existing valid certificates will be reused.
    """
    if cert_dir is None:
        cert_dir = Path.home() / ".qtrader" / "ssl"

    cert_path = cert_dir / "localhost.pem"
    key_path = cert_dir / "localhost-key.pem"

    # Check if certificates already exist and are valid
    if cert_path.exists() and key_path.exists():
        try:
            # Load and check certificate validity
            with open(cert_path, "rb") as f:
                cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)

            # Check if certificate is still valid
            now = datetime.datetime.now(datetime.timezone.utc)
            if cert.not_valid_before_utc <= now <= cert.not_valid_after_utc:
                logger.info(
                    "ssl_generator.using_existing_certificates",
                    cert_path=str(cert_path),
                    expires=cert.not_valid_after_utc.isoformat(),
                )
                return cert_path, key_path
            else:
                logger.warning(
                    "ssl_generator.certificate_expired",
                    expired_at=cert.not_valid_after_utc.isoformat(),
                )
        except Exception as e:
            logger.warning("ssl_generator.certificate_validation_failed", error=str(e))

    # Generate new certificates
    logger.info("ssl_generator.generating_new_certificates")
    return generate_self_signed_cert(cert_dir)
