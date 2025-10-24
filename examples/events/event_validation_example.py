"""
Example: Creating and Validating Events with the New System

Demonstrates:
1. Creating validated events (PriceBarEvent, CorporateActionEvent)
2. Creating control events (BarCloseEvent, BacktestStartedEvent)
3. Automatic envelope + payload validation
4. Type-safe Decimal and datetime handling
5. Error handling with detailed context
"""

from datetime import timezone
from decimal import Decimal

from qtrader.events import BacktestStartedEvent, BarCloseEvent, CorporateActionEvent, PriceBarEvent


def example_price_bar_event():
    """Create a validated price bar event."""
    print("\n=== PriceBarEvent Example ===")

    # Create event with type-safe fields
    event = PriceBarEvent(
        # Envelope fields (auto-generated/defaulted)
        source_service="data_service",
        # correlation_id must be UUID format (or None)
        # Domain fields (validated against bar.v1.json)
        symbol="AAPL",
        asset_class="equity",
        interval="1d",
        timestamp="2024-01-01T00:00:00Z",
        open=Decimal("150.00"),  # Accepts Decimal
        high="155.00",  # Also accepts string (auto-converted)
        low=Decimal("149.00"),
        close=Decimal("154.50"),
        volume=1_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="algoseek",
    )

    print(f"✓ Event created: {event.event_id}")
    print(f"  Symbol: {event.symbol}")
    print(f"  Close: {event.close} (type: {type(event.close).__name__})")
    print(f"  Occurred at: {event.occurred_at} (UTC: {event.occurred_at.tzinfo})")
    print(f"  Frozen: {event.model_config.get('frozen')}")

    # Try to modify (will fail - immutable)
    try:
        event.close = Decimal("160.00")
    except Exception as e:
        print(f"✓ Immutability enforced: {type(e).__name__}")

    # Serialize for JSON Schema validation
    data = event.model_dump()
    print(f"  Serialized close: {data['close']} (type: {type(data['close']).__name__})")

    return event


def example_corporate_action_event():
    """Create a validated corporate action event."""
    print("\n=== CorporateActionEvent Example ===")

    # Apple 4-for-1 split (2020-08-31)
    event = CorporateActionEvent(
        source_service="data_service",
        symbol="AAPL",
        asset_class="equity",
        action_type="split",
        announcement_date="2020-07-30",
        ex_date="2020-08-31",
        effective_date="2020-08-31",
        source="algoseek",
        split_from=1,
        split_to=4,
        split_ratio=Decimal("0.25"),
        price_adjustment_factor=Decimal("0.25"),
        volume_adjustment_factor=Decimal("4.0"),
    )

    print(f"✓ Event created: {event.event_id}")
    print(f"  {event.symbol} {event.action_type}: {event.split_from}-for-{event.split_to}")
    print(f"  Split ratio: {event.split_ratio}")
    print(f"  Ex-date: {event.ex_date}")

    return event


def example_control_event():
    """Create a control event (no payload validation)."""
    print("\n=== ControlEvent Example ===")

    # Backtest lifecycle event
    event = BacktestStartedEvent(
        source_service="backtest_engine",
        config={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 100_000,
            "strategy": "momentum",
        },
    )

    print(f"✓ Event created: {event.event_id}")
    print(f"  Type: {event.event_type}")
    print(f"  Config: {event.config}")
    print(f"  No payload validation required (ControlEvent)")

    # Barrier event
    barrier = BarCloseEvent(
        source_service="data_service",
    )
    print(f"✓ Barrier created: {barrier.event_id} ({barrier.event_type})")

    return event


def example_validation_error():
    """Demonstrate validation error with detailed context."""
    print("\n=== Validation Error Example ===")

    try:
        # Missing required fields
        PriceBarEvent(
            symbol="AAPL",
            asset_class="equity",
            interval="1d",
            timestamp="2024-01-01T00:00:00Z",
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.50"),
            # Missing: volume, cumulative_price_factor, cumulative_volume_factor, source
        )
    except Exception as e:
        print(f"✗ Validation failed: {type(e).__name__}")
        error_msg = str(e)
        if "payload validation failed" in error_msg:
            print(f"  → Payload validation error")
            if "Path:" in error_msg:
                print(f"  → Error includes path context")
            print(f"  → Message snippet: {error_msg[:200]}...")
        else:
            print(f"  → Pydantic validation error (field-level)")

    try:
        # Invalid decimal format
        PriceBarEvent(
            symbol="AAPL",
            asset_class="equity",
            interval="1d",
            timestamp="2024-01-01T00:00:00Z",
            open="not-a-number",  # Invalid
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.50"),
            volume=1_000_000,
            adjusted=False,
            cumulative_price_factor=Decimal("1.0"),
            cumulative_volume_factor=Decimal("1.0"),
            source="algoseek",
        )
    except Exception as e:
        print(f"✗ Invalid decimal: {type(e).__name__}")
        print(f"  → Pydantic caught invalid input before schema validation")


def example_timezone_handling():
    """Demonstrate UTC timezone handling."""
    print("\n=== Timezone Handling Example ===")

    # All timestamps converted to UTC
    event = BarCloseEvent(source_service="data_service")

    print(f"✓ Timestamp is UTC: {event.occurred_at.tzinfo == timezone.utc}")
    print(f"  Occurred at: {event.occurred_at}")

    # Serialize to RFC3339 with Z
    data = event.model_dump()
    print(f"  Serialized: {data['occurred_at']}")
    print(f"  ✓ Ends with 'Z': {data['occurred_at'].endswith('Z')}")


def example_version_upgrade():
    """Demonstrate version management."""
    print("\n=== Version Management Example ===")

    # Create v1 event
    event_v1 = PriceBarEvent(
        symbol="AAPL",
        asset_class="equity",
        interval="1d",
        timestamp="2024-01-01T00:00:00Z",
        open=Decimal("150.00"),
        high=Decimal("155.00"),
        low=Decimal("149.00"),
        close=Decimal("154.50"),
        volume=1_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="algoseek",
    )

    print(f"✓ Created v{event_v1.event_version} event")
    print(f"  Schema: bar.v{event_v1.event_version}.json")
    print(f"  (To upgrade to v2: just change event_version=2)")
    print(f"  (Schema loader automatically uses bar.v2.json)")


if __name__ == "__main__":
    print("=" * 70)
    print("QTrader Event System Examples")
    print("=" * 70)

    example_price_bar_event()
    example_corporate_action_event()
    example_control_event()
    example_validation_error()
    example_timezone_handling()
    example_version_upgrade()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
