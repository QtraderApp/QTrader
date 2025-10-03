"""
Integration test: Data pipeline from adapter to bars.

Tests the data loading workflow:
    Data Files → Adapter → Bar Objects → Validation

This is a placeholder for when we have adapter integration tests.
Currently, adapter functionality is tested in unit tests.
"""

import pytest


@pytest.mark.skip(reason="Adapter integration tests require full data loading implementation")
def test_data_pipeline_placeholder():
    """
    Placeholder for data pipeline integration tests.

    Future tests will cover:
    - Load real fixture data through adapter
    - Verify Bar normalization (Decimal prices, timezone-aware timestamps)
    - Test date range filtering
    - Test symbol filtering
    - Test bar sorting (by symbol, then timestamp)
    - Verify adjustment events loaded separately
    """
    pass


@pytest.mark.skip(reason="Multi-dataset integration tests for future phases")
def test_multi_dataset_alignment_placeholder():
    """
    Placeholder for multi-dataset integration tests.

    Future tests will cover:
    - Load primary dataset (OHLC bars)
    - Load auxiliary datasets (fundamentals, sentiment, etc.)
    - Test data alignment strategies (forward-fill, join)
    - Verify missing data handling
    """
    pass
