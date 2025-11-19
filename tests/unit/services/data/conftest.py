"""Conftest for data service unit tests."""

import pytest

from qtrader.services.data.adapters.resolver import DataSourceResolver


@pytest.fixture
def test_resolver():
    """Create DataSourceResolver pointing to test fixtures."""
    return DataSourceResolver(config_path="tests/fixtures/config/data_sources.yaml")
