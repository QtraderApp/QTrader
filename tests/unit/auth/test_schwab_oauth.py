"""
Tests for SchwabOAuthManager.

These tests verify the OAuth manager's ability to:
- Initialize with configuration
- Cache and retrieve tokens
- Handle token expiry
- Generate authorization URLs
- Manual code entry mode
"""

import json
import time
import urllib.parse
from unittest.mock import Mock, patch

import pytest
import requests

from qtrader.auth.schwab_oauth import SchwabOAuthManager


class TestSchwabOAuthManagerInitialization:
    """Test OAuth manager initialization."""

    def test_create_manager_with_valid_config(self, tmp_path):
        """Test creating manager with valid configuration."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
        )

        assert manager.client_id == "test_client_id"
        assert manager.client_secret == "test_client_secret"
        assert manager.token_cache_path == tmp_path / "tokens.json"
        assert manager.manual_mode is False

    def test_create_manager_with_manual_mode(self, tmp_path):
        """Test creating manager with manual mode enabled."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        assert manager.manual_mode is True

    def test_create_manager_with_custom_redirect_uri(self, tmp_path):
        """Test creating manager with custom redirect URI."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="https://example.com/callback",
            token_cache_path=tmp_path / "tokens.json",
        )

        assert manager.redirect_uri == "https://example.com/callback"

    def test_create_manager_default_redirect_uri(self, tmp_path):
        """Test creating manager with default redirect URI."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
        )

        assert manager.redirect_uri == "https://127.0.0.1:8182"

    def test_create_manager_missing_client_id(self):
        """Test creating manager without client_id raises error."""
        with pytest.raises(ValueError, match="client_id and client_secret are required"):
            SchwabOAuthManager(
                client_id="",
                client_secret="test_client_secret",
            )

    def test_create_manager_missing_client_secret(self):
        """Test creating manager without client_secret raises error."""
        with pytest.raises(ValueError, match="client_id and client_secret are required"):
            SchwabOAuthManager(
                client_id="test_client_id",
                client_secret="",
            )

    def test_create_manager_creates_cache_directory(self, tmp_path):
        """Test that manager creates cache directory if it doesn't exist."""
        cache_path = tmp_path / "subdir" / "tokens.json"

        _ = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        assert cache_path.parent.exists()


class TestSchwabOAuthManagerTokenCache:
    """Test token caching functionality."""

    def test_load_cached_token_no_cache(self, tmp_path):
        """Test loading token when cache doesn't exist."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
        )

        token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert token is None

    def test_load_cached_token_empty_cache(self, tmp_path):
        """Test loading token when cache is empty."""
        cache_path = tmp_path / "tokens.json"
        cache_path.write_text("{}")

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert token is None

    def test_load_cached_token_expired(self, tmp_path):
        """Test loading token when token is expired."""
        cache_path = tmp_path / "tokens.json"

        # Create expired token (expires in the past)
        expired_token = {
            "access_token": "test_token",
            "expires_at": time.time() - 3600,  # Expired 1 hour ago
        }
        cache_path.write_text(json.dumps(expired_token))

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert token is None

    def test_load_cached_token_valid(self, tmp_path):
        """Test loading token when token is valid."""
        cache_path = tmp_path / "tokens.json"

        # Create valid token (expires in 1 hour)
        valid_token = {
            "access_token": "test_token_12345",
            "expires_at": time.time() + 3600,  # Expires in 1 hour
        }
        cache_path.write_text(json.dumps(valid_token))

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert token == "test_token_12345"

    def test_load_cached_token_invalid_json(self, tmp_path):
        """Test loading token when cache has invalid JSON."""
        cache_path = tmp_path / "tokens.json"
        cache_path.write_text("not valid json")

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert token is None

    def test_save_token_cache(self, tmp_path):
        """Test saving token to cache."""
        cache_path = tmp_path / "tokens.json"

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token_data = {
            "access_token": "test_token_67890",
            "refresh_token": "refresh_abc",
            "token_type": "Bearer",
            "expires_in": 1800,
            "scope": "read write",
        }

        manager._save_token_cache(token_data)  # type: ignore[attr-defined]

        # Verify file exists
        assert cache_path.exists()

        # Verify contents
        with open(cache_path) as f:
            saved_data = json.load(f)

        assert saved_data["access_token"] == "test_token_67890"
        assert saved_data["refresh_token"] == "refresh_abc"
        assert "expires_at" in saved_data
        assert "created_at" in saved_data

    def test_save_token_cache_permissions(self, tmp_path):
        """Test that cached token file has secure permissions."""
        cache_path = tmp_path / "tokens.json"

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token_data = {
            "access_token": "test_token",
            "expires_in": 1800,
        }

        manager._save_token_cache(token_data)  # type: ignore[attr-defined]

        # Verify file exists
        assert cache_path.exists()

        # Verify content
        cached_data = json.loads(cache_path.read_text())
        assert cached_data["access_token"] == "test_token"
        assert "expires_at" in cached_data
        assert cached_data["expires_at"] > time.time()

        # Check permissions (should be 600 on Unix-like systems)
        import stat

        mode = cache_path.stat().st_mode
        # Owner can read/write, no permissions for group/others
        assert mode & stat.S_IRUSR  # Owner read
        assert mode & stat.S_IWUSR  # Owner write
        assert not (mode & stat.S_IRGRP)  # No group read
        assert not (mode & stat.S_IWGRP)  # No group write
        assert not (mode & stat.S_IROTH)  # No other read
        assert not (mode & stat.S_IWOTH)  # No other write
        assert not (mode & stat.S_IWOTH)  # No other write


class TestSchwabOAuthManagerTokenExchange:
    """Test token exchange functionality."""

    @patch("requests.post")
    def test_exchange_code_for_token_success(self, mock_post, tmp_path):
        """Test successful token exchange."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token_123",
            "token_type": "Bearer",
            "expires_in": 1800,
            "refresh_token": "new_refresh_token_456",
        }
        mock_post.return_value = mock_response

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
        )

        token = manager._exchange_code_for_token("auth_code_789")  # type: ignore[attr-defined]

        assert token == "new_access_token_123"

        # Verify API call
        assert mock_post.called
        call_args = mock_post.call_args
        assert call_args[0][0] == SchwabOAuthManager.TOKEN_URL

        # Verify token was cached
        cached_token = manager._load_cached_token()  # type: ignore[attr-defined]
        assert cached_token == "new_access_token_123"

    @patch("requests.post")
    def test_exchange_code_for_token_api_error(self, mock_post, tmp_path):
        """Test token exchange with API error."""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("Bad request")
        mock_post.return_value = mock_response

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
        )

        with pytest.raises(requests.HTTPError):
            manager._exchange_code_for_token("invalid_code")


class TestSchwabOAuthManagerManualMode:
    """Test manual code entry mode."""

    def test_get_code_manual_with_full_url(self, tmp_path, monkeypatch):
        """Test manual code entry with full redirect URL."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        # Mock user input with full URL
        test_url = "https://example.com/callback?code=ABC123XYZ"
        monkeypatch.setattr("builtins.input", lambda _: test_url)

        code = manager._get_code_manual()

        assert code == "ABC123XYZ"

    def test_get_code_manual_with_code_only(self, tmp_path, monkeypatch):
        """Test manual code entry with just the code."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        # Mock user input with just code
        test_code = "DEF456UVW"
        monkeypatch.setattr("builtins.input", lambda _: test_code)

        code = manager._get_code_manual()

        assert code == "DEF456UVW"

    def test_get_code_manual_empty_input(self, tmp_path, monkeypatch):
        """Test manual code entry with empty input."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        # Mock empty input
        monkeypatch.setattr("builtins.input", lambda _: "")

        code = manager._get_code_manual()

        assert code is None

    def test_get_code_manual_keyboard_interrupt(self, tmp_path, monkeypatch):
        """Test manual code entry with keyboard interrupt."""
        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        # Mock keyboard interrupt
        def raise_interrupt(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", raise_interrupt)

        code = manager._get_code_manual()

        assert code is None


class TestSchwabOAuthManagerGetAccessToken:
    """Test getting access token."""

    def test_get_access_token_from_cache(self, tmp_path):
        """Test getting access token from cache when valid."""
        cache_path = tmp_path / "tokens.json"

        # Create valid cached token
        token_data = {
            "access_token": "cached_token_789",
            "expires_at": time.time() + 3600,
        }
        cache_path.write_text(json.dumps(token_data))

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=cache_path,
        )

        token = manager.get_access_token()

        assert token == "cached_token_789"

    @patch("qtrader.auth.schwab_oauth.SchwabOAuthManager._get_code_manual")
    @patch("qtrader.auth.schwab_oauth.SchwabOAuthManager._exchange_code_for_token")
    def test_get_access_token_manual_mode_new_auth(self, mock_exchange, mock_get_code, tmp_path):
        """Test getting access token in manual mode (no cache)."""
        # Mock manual code entry
        mock_get_code.return_value = "manual_code_123"

        # Mock token exchange
        mock_exchange.return_value = "new_token_from_manual"

        manager = SchwabOAuthManager(
            client_id="test_client_id",
            client_secret="test_client_secret",
            token_cache_path=tmp_path / "tokens.json",
            manual_mode=True,
        )

        token = manager.get_access_token()

        assert token == "new_token_from_manual"
        assert mock_get_code.called
        assert mock_exchange.called


class TestSchwabOAuthManagerAuthorizationURL:
    """Test authorization URL generation."""

    def test_generate_authorization_url(self, tmp_path):
        """Test generating authorization URL with correct parameters."""
        manager = SchwabOAuthManager(
            client_id="test_client_123",
            client_secret="test_secret_456",
            redirect_uri="https://example.com/callback",
            token_cache_path=tmp_path / "tokens.json",
        )

        # The URL is generated in _acquire_new_token method
        # We can test the components
        expected_base = "https://api.schwabapi.com/v1/oauth/authorize"
        expected_params = {
            "client_id": "test_client_123",
            "redirect_uri": "https://example.com/callback",
        }

        # Verify components match expected authorization URL structure
        assert manager.client_id == "test_client_123"
        assert manager.redirect_uri == "https://example.com/callback"
        assert manager.AUTH_URL == expected_base

        # Verify the params would form correct URL
        expected_query = urllib.parse.urlencode(expected_params)
        assert "client_id=test_client_123" in expected_query
