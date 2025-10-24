#!/usr/bin/env python3
"""
Example: Local Timestamps for Market Session Analysis

Demonstrates:
- timestamp (UTC, always required)
- timestamp_local (RFC3339 with offset, optional)
- timezone (IANA timezone, optional)

Use cases:
- Equity markets: session-specific analysis (market open/close, intraday patterns)
- Crypto/forex: omit local fields (24/7 global markets)
"""

from datetime import datetime, timezone
from decimal import Decimal

from qtrader.events.events import PriceBarEvent


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print("=" * 70)


def example_equity_market_session():
    """Example: NYSE equity bar with local session time."""
    print_section("Example 1: NYSE Equity Bar (9:30 AM ET Market Open)")

    # Create bar for AAPL at market open
    # Note: During EST (winter), offset is -05:00
    #       During EDT (summer), offset is -04:00
    event = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        # Domain fields
        symbol="AAPL",
        asset_class="equity",
        interval="1m",
        # Timestamp fields
        timestamp="2024-03-10T14:30:00Z",  # UTC: 2:30 PM
        timestamp_local="2024-03-10T09:30:00-05:00",  # EST: 9:30 AM (market open)
        timezone="America/New_York",
        # OHLCV
        open=Decimal("150.00"),
        high=Decimal("150.50"),
        low=Decimal("149.80"),
        close=Decimal("150.25"),
        volume=1_500_000,
        # Adjustment factors
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="nyse",
    )

    print(f"✓ Symbol: {event.symbol}")
    print(f"  Timestamp (UTC):   {event.timestamp}")
    print(f"  Timestamp (Local): {event.timestamp_local}")
    print(f"  Timezone:          {event.timezone}")
    print(f"  → Market open time preserved without DST ambiguity")


def example_dst_transition():
    """Example: DST transition handling."""
    print_section("Example 2: DST Transition (EST → EDT)")

    # Before DST: EST = UTC-5
    pre_dst = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        symbol="MSFT",
        asset_class="equity",
        interval="1d",
        timestamp="2024-03-09T21:00:00Z",  # UTC
        timestamp_local="2024-03-09T16:00:00-05:00",  # 4 PM EST (market close)
        timezone="America/New_York",
        open=Decimal("400.00"),
        high=Decimal("405.00"),
        low=Decimal("399.00"),
        close=Decimal("404.50"),
        volume=25_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="nasdaq",
    )

    # After DST: EDT = UTC-4
    post_dst = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        symbol="MSFT",
        asset_class="equity",
        interval="1d",
        timestamp="2024-03-11T20:00:00Z",  # UTC (note: 1 hour earlier)
        timestamp_local="2024-03-11T16:00:00-04:00",  # 4 PM EDT (market close)
        timezone="America/New_York",
        open=Decimal("405.00"),
        high=Decimal("408.00"),
        low=Decimal("403.00"),
        close=Decimal("407.25"),
        volume=28_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="nasdaq",
    )

    print(f"Pre-DST  (Mar 9):  UTC={pre_dst.timestamp}, Local={pre_dst.timestamp_local}")
    print(f"Post-DST (Mar 11): UTC={post_dst.timestamp}, Local={post_dst.timestamp_local}")
    print(f"→ Both represent 4 PM market close, offset automatically handles DST")


def example_global_exchanges():
    """Example: Different exchanges with different timezones."""
    print_section("Example 3: Global Exchanges (NYSE, LSE, TSE)")

    # NYSE: New York (EST/EDT)
    nyse_bar = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        symbol="AAPL",
        asset_class="equity",
        interval="1h",
        timestamp="2024-01-15T14:30:00Z",  # UTC
        timestamp_local="2024-01-15T09:30:00-05:00",  # 9:30 AM EST
        timezone="America/New_York",
        open=Decimal("185.00"),
        high=Decimal("186.00"),
        low=Decimal("184.50"),
        close=Decimal("185.75"),
        volume=5_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="nyse",
    )

    # LSE: London (GMT/BST)
    lse_bar = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        symbol="BARC.L",
        asset_class="equity",
        interval="1h",
        timestamp="2024-01-15T08:00:00Z",  # UTC (winter = GMT)
        timestamp_local="2024-01-15T08:00:00+00:00",  # 8 AM GMT
        timezone="Europe/London",
        open=Decimal("1.45"),
        high=Decimal("1.47"),
        low=Decimal("1.44"),
        close=Decimal("1.46"),
        volume=10_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="lse",
    )

    # TSE: Tokyo (JST, no DST)
    tse_bar = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="market_data_service",
        symbol="7203.T",  # Toyota
        asset_class="equity",
        interval="1h",
        timestamp="2024-01-15T00:00:00Z",  # UTC
        timestamp_local="2024-01-15T09:00:00+09:00",  # 9 AM JST
        timezone="Asia/Tokyo",
        open=Decimal("2450.00"),
        high=Decimal("2465.00"),
        low=Decimal("2445.00"),
        close=Decimal("2460.00"),
        volume=3_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="tse",
    )

    print(f"NYSE (AAPL):  {nyse_bar.timestamp_local} ({nyse_bar.timezone})")
    print(f"LSE  (BARC):  {lse_bar.timestamp_local} ({lse_bar.timezone})")
    print(f"TSE  (7203):  {tse_bar.timestamp_local} ({tse_bar.timezone})")
    print(f"→ Each exchange has its own local session time")


def example_crypto_24x7():
    """Example: 24/7 crypto market (no local timestamps needed)."""
    print_section("Example 4: Crypto (24/7, No Local Timestamps)")

    # Crypto trades 24/7, no "market session" concept
    crypto_bar = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="crypto_data_service",
        symbol="BTC-USD",
        asset_class="crypto",
        interval="1h",
        timestamp="2024-01-15T12:00:00Z",  # UTC only
        timestamp_local=None,  # Not relevant for 24/7 markets
        timezone=None,
        open=Decimal("42500.00"),
        high=Decimal("42800.00"),
        low=Decimal("42300.00"),
        close=Decimal("42650.00"),
        volume=150_000_000,  # High volume
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="binance",
    )

    print(f"✓ Symbol: {crypto_bar.symbol}")
    print(f"  Timestamp (UTC):   {crypto_bar.timestamp}")
    print(f"  Timestamp (Local): {crypto_bar.timestamp_local}")
    print(f"  Timezone:          {crypto_bar.timezone}")
    print(f"  → 24/7 market doesn't need local time")


def example_serialization():
    """Example: Serialization with/without local fields."""
    print_section("Example 5: Serialization (Wire Format)")

    # Equity with local fields
    equity = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="test",
        symbol="TSLA",
        asset_class="equity",
        interval="5m",
        timestamp="2024-01-15T14:35:00Z",
        timestamp_local="2024-01-15T09:35:00-05:00",
        timezone="America/New_York",
        open=Decimal("250.00"),
        high=Decimal("251.00"),
        low=Decimal("249.50"),
        close=Decimal("250.75"),
        volume=2_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="nasdaq",
    )

    # Crypto without local fields
    crypto = PriceBarEvent(
        event_type="bar",
        event_version=1,
        occurred_at=datetime.now(timezone.utc),
        source_service="test",
        symbol="ETH-USD",
        asset_class="crypto",
        interval="5m",
        timestamp="2024-01-15T14:35:00Z",
        open=Decimal("2300.00"),
        high=Decimal("2310.00"),
        low=Decimal("2295.00"),
        close=Decimal("2305.00"),
        volume=50_000_000,
        adjusted=False,
        cumulative_price_factor=Decimal("1.0"),
        cumulative_volume_factor=Decimal("1.0"),
        source="coinbase",
    )

    equity_json = equity.model_dump(exclude_none=True)
    crypto_json = crypto.model_dump(exclude_none=True)

    print("Equity serialization (with local fields):")
    print(f"  timestamp:       {equity_json['timestamp']}")
    print(f"  timestamp_local: {equity_json['timestamp_local']}")
    print(f"  timezone:        {equity_json['timezone']}")

    print("\nCrypto serialization (without local fields):")
    print(f"  timestamp:       {crypto_json['timestamp']}")
    print(f"  timestamp_local: {'timestamp_local' in crypto_json}")
    print(f"  timezone:        {'timezone' in crypto_json}")
    print(f"  → Crypto omits optional fields (exclude_none=True)")


def example_session_analysis():
    """Example: SQL query benefits with local timestamps."""
    print_section("Example 6: Session Analysis Use Case")

    print("Without timestamp_local (hard):")
    print("""
    -- Complex timezone conversion required
    SELECT
      DATE(CONVERT_TZ(timestamp, '+00:00', 'America/New_York')) AS trading_day,
      HOUR(CONVERT_TZ(timestamp, '+00:00', 'America/New_York')) AS session_hour,
      AVG(close) AS avg_close
    FROM bars
    WHERE symbol = 'AAPL'
    GROUP BY 1, 2;
    """)

    print("With timestamp_local (simple):")
    print("""
    -- No timezone math needed!
    SELECT
      DATE(timestamp_local) AS trading_day,
      HOUR(timestamp_local) AS session_hour,
      AVG(close) AS avg_close
    FROM bars
    WHERE symbol = 'AAPL' AND timezone = 'America/New_York'
    GROUP BY 1, 2;
    """)

    print("→ timestamp_local makes session analysis trivial")


def main():
    """Run all examples."""
    print("=" * 70)
    print("QTrader Event System: Local Timestamp Examples")
    print("=" * 70)

    example_equity_market_session()
    example_dst_transition()
    example_global_exchanges()
    example_crypto_24x7()
    example_serialization()
    example_session_analysis()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
