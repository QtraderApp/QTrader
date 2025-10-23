#!/usr/bin/env python3
"""QTrader Full Run Example - Registry & Configuration."""

from pathlib import Path

from qtrader.engine.config import load_backtest_config
from qtrader.libraries.registry import IndicatorRegistry, StrategyRegistry


def display_indicators_info(registry):
    """Display discovered indicators."""
    print("=" * 80)
    print("INDICATOR REGISTRY")
    print("=" * 80)

    indicators = registry.list_components()

    if not indicators:
        print("\nNo indicators discovered!")
        return

    buildin_indicators = []
    custom_indicators = []

    for name in sorted(indicators.keys()):
        metadata = registry.get_metadata(name)
        source_type = metadata.get("source_type", "unknown")

        info = {
            "name": name,
            "class": indicators[name].__name__,
            "module": metadata.get("module_path", "unknown"),
        }

        if source_type == "buildin":
            buildin_indicators.append(info)
        else:
            custom_indicators.append(info)

    if buildin_indicators:
        print(f"\nBuilt-in Indicators ({len(buildin_indicators)}):")
        for info in buildin_indicators:
            print(f"   {info['name']:20s} ({info['class']})")

    if custom_indicators:
        print(f"\nCustom Indicators ({len(custom_indicators)}):")
        for info in custom_indicators:
            print(f"   {info['name']:20s} ({info['class']})")

    print(f"\nTotal Indicators: {len(indicators)}")
    print("=" * 80)


def display_strategies_info(registry):
    """Display discovered strategies."""
    print("=" * 80)
    print("STRATEGY REGISTRY")
    print("=" * 80)

    strategies = registry.list_components()

    if not strategies:
        print("\nNo strategies discovered!")
        return

    buildin_strategies = []
    custom_strategies = []

    for name in sorted(strategies.keys()):
        metadata = registry.get_metadata(name)
        source_type = metadata.get("source_type", "unknown")

        info = {
            "name": name,
            "class": strategies[name].__name__,
            "module": metadata.get("module_path", "unknown"),
        }

        if source_type == "buildin":
            buildin_strategies.append(info)
        else:
            custom_strategies.append(info)

    if buildin_strategies:
        print(f"\nBuilt-in Strategies ({len(buildin_strategies)}):")
        for info in buildin_strategies:
            print(f"   {info['name']:20s} ({info['class']})")

    if custom_strategies:
        print(f"\nCustom Strategies ({len(custom_strategies)}):")
        for info in custom_strategies:
            print(f"   {info['name']:20s} ({info['class']})")

    print(f"\nTotal Strategies: {len(strategies)}")
    print("=" * 80)


def display_config_info(config):
    """Display loaded configuration details."""
    print("=" * 80)
    print("BACKTEST CONFIGURATION")
    print("=" * 80)

    print("\nBacktest Parameters:")
    print(f"   Start Date:      {config.start_date.strftime('%Y-%m-%d')}")
    print(f"   End Date:        {config.end_date.strftime('%Y-%m-%d')}")
    print(f"   Initial Equity:  ${config.initial_equity:,.2f}")

    print("\nData Sources:")
    for source in config.data.sources:
        print(f"   * {source.name}")
        print(f"     Universe: {', '.join(source.universe)}")

    print(f"\n   Total Symbols: {len(config.all_symbols)}")

    print("\nStrategies:")
    for i, strategy in enumerate(config.strategies, 1):
        print(f"   {i}. {strategy.strategy_id}")

    print("\n" + "=" * 80)


def main():
    """Load configuration and demonstrate registries."""

    print("=" * 80)
    print("QTrader - Registry & Configuration Demo")
    print("=" * 80)
    print()

    print("Step 1: Discovering Indicators...")
    print()

    indicator_registry = IndicatorRegistry()

    buildin_indicator_path = Path("src/qtrader/libraries/indicators/buildin")
    custom_indicator_path = Path("my_library/indicators")

    indicator_counts = indicator_registry.discover(
        buildin_path=buildin_indicator_path,
        custom_paths=[custom_indicator_path] if custom_indicator_path.exists() else [],
    )

    print(f"Discovered {indicator_counts['buildin']} built-in indicators")
    print(f"Discovered {indicator_counts['custom']} custom indicators")
    print()

    display_indicators_info(indicator_registry)
    print()

    print("Step 2: Discovering Strategies...")
    print()

    strategy_registry = StrategyRegistry()

    buildin_strategy_path = Path("src/qtrader/libraries/strategies/buildin")
    custom_strategy_path = Path("my_library/strategies")

    strategy_counts = strategy_registry.discover(
        buildin_path=buildin_strategy_path,
        custom_paths=[custom_strategy_path] if custom_strategy_path.exists() else [],
    )

    print(f"Discovered {strategy_counts['buildin']} built-in strategies")
    print(f"Discovered {strategy_counts['custom']} custom strategies")
    print()

    display_strategies_info(strategy_registry)
    print()

    print("Step 3: Loading Configuration...")
    print()

    try:
        config = load_backtest_config(Path("config/portfolio.yaml"))
        print("Configuration loaded successfully!")
        print()

        display_config_info(config)

        print("\nStep 4: Demonstrating Component Usage...")
        print("=" * 80)

        # Demonstrate indicator creation
        if "sma" in indicator_registry:
            SMA = indicator_registry.get("sma")
            sma = SMA(period=20)
            print(f"Created SMA(20): {type(sma).__name__}")

        if "bollingerbands" in indicator_registry:
            BB = indicator_registry.get("bollingerbands")
            bb = BB(period=20, num_std=2.0)
            print(f"Created BollingerBands(20, 2.0): {type(bb).__name__}")

        # Demonstrate strategy creation
        print()

        if "buyandholdstrategy" in strategy_registry:
            from my_library.strategies import BuyAndHoldConfig

            BuyAndHold = strategy_registry.get("buyandholdstrategy")
            bah_config = BuyAndHoldConfig()
            _ = BuyAndHold(bah_config)  # Demonstrate instantiation
            print(f"✓ {bah_config.display_name} v{bah_config.version}")
            print(f"  {bah_config.description}")
            print(f"  Warmup: {bah_config.warmup_bars} bars")

        print()

        if "bollingerbreakoutstrategy" in strategy_registry:
            from my_library.strategies import BollingerBreakoutConfig

            BollingerBreakout = strategy_registry.get("bollingerbreakoutstrategy")
            bb_config = BollingerBreakoutConfig(
                name="bb_breakout",
                bb_period=20,
                bb_num_std=2.0,
            )
            _ = BollingerBreakout(bb_config)  # Demonstrate instantiation
            print(f"✓ {bb_config.display_name} v{bb_config.version}")
            print(f"  {bb_config.description[:60]}...")
            print(f"  Warmup: {bb_config.warmup_bars} bars | BB Period: {bb_config.bb_period}")

        print("\n" + "=" * 80)
        print("\nRegistry demonstration complete!")
        print(f"  {len(indicator_registry.list_names())} indicators available")
        print(f"  {len(strategy_registry.list_names())} strategies available")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
