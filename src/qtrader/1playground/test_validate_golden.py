"""
Golden Output Validation Test

This test validates that the refactored architecture produces identical output
to the original implementation. It loads the golden output and compares it
against the new implementation.

Run this after refactoring to ensure correctness.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import List


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str):
        """Add an error (test fails)."""
        self.passed = False
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning (test passes but something is noteworthy)."""
        self.warnings.append(message)

    def report(self):
        """Print validation report."""
        print("\n" + "=" * 100)
        print("VALIDATION REPORT")
        print("=" * 100)

        if self.passed:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ TESTS FAILED")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.passed and not self.warnings:
            print("\n🎉 Perfect match! Refactoring is correct.")

        print("=" * 100)

        return self.passed


def load_golden_output(path: Path) -> dict:
    """Load golden output from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compare_bars(
    golden_bars: List[dict],
    test_bars: List[dict],
    mode: str,
    result: ValidationResult,
    tolerance: float = 0.01,  # $0.01 for prices, exact for volume
):
    """Compare two lists of bars."""

    if len(golden_bars) != len(test_bars):
        result.add_error(f"{mode}: Bar count mismatch. Expected {len(golden_bars)}, got {len(test_bars)}")
        return

    for i, (golden, test) in enumerate(zip(golden_bars, test_bars)):
        date = golden["trade_datetime"]

        # Compare each field
        if golden["trade_datetime"] != test["trade_datetime"]:
            result.add_error(
                f"{mode} bar {i}: Date mismatch. Expected {golden['trade_datetime']}, got {test['trade_datetime']}"
            )
            continue

        # Compare OHLC (with tolerance for floating point)
        for field in ["open", "high", "low", "close"]:
            golden_val = golden[field]
            test_val = test[field]
            diff = abs(golden_val - test_val)

            if diff > tolerance:
                result.add_error(
                    f"{mode} {date}: {field} mismatch. Expected {golden_val:.2f}, got {test_val:.2f}, diff={diff:.4f}"
                )

        # Compare volume (exact match required)
        if golden["volume"] != test["volume"]:
            result.add_error(f"{mode} {date}: Volume mismatch. Expected {golden['volume']}, got {test['volume']}")

        # Compare dividend if present
        golden_div = golden.get("dividend")
        test_div = test.get("dividend")

        if golden_div is not None and test_div is None:
            result.add_error(f"{mode} {date}: Expected dividend {golden_div}, got None")
        elif golden_div is None and test_div is not None:
            result.add_error(f"{mode} {date}: Expected no dividend, got {test_div}")
        elif golden_div is not None and test_div is not None:
            # Compare as Decimal strings
            golden_div_decimal = Decimal(golden_div)
            test_div_decimal = Decimal(test_div)
            diff = abs(golden_div_decimal - test_div_decimal)

            if diff > Decimal("0.0001"):  # 1/100th of a cent tolerance
                result.add_error(
                    f"{mode} {date}: Dividend mismatch. Expected {golden_div}, got {test_div}, diff={diff}"
                )


def validate_against_golden(test_data: dict, golden_path: Path) -> bool:
    """
    Validate test data against golden output.

    Args:
        test_data: Dictionary with same structure as golden output
        golden_path: Path to golden output JSON file

    Returns:
        True if validation passes, False otherwise
    """
    result = ValidationResult()

    # Load golden output
    print("=" * 100)
    print("LOADING GOLDEN OUTPUT")
    print("=" * 100)

    try:
        golden_data = load_golden_output(golden_path)
        print(f"✅ Loaded golden output from: {golden_path}")
    except FileNotFoundError:
        result.add_error(f"Golden output file not found: {golden_path}")
        result.report()
        return False
    except json.JSONDecodeError as e:
        result.add_error(f"Invalid JSON in golden output: {e}")
        result.report()
        return False

    # Validate metadata
    print("\n" + "=" * 100)
    print("VALIDATING METADATA")
    print("=" * 100)

    golden_meta = golden_data["metadata"]
    test_meta = test_data.get("metadata", {})

    if test_meta.get("symbol") != golden_meta["symbol"]:
        result.add_error(f"Symbol mismatch. Expected {golden_meta['symbol']}, got {test_meta.get('symbol')}")
    else:
        print(f"✅ Symbol: {golden_meta['symbol']}")

    # Validate series
    print("\n" + "=" * 100)
    print("VALIDATING SERIES")
    print("=" * 100)

    for mode in ["unadjusted", "adjusted", "total_return"]:
        print(f"\nValidating {mode}...")

        if mode not in test_data.get("series", {}):
            result.add_error(f"Missing series: {mode}")
            continue

        golden_series = golden_data["series"][mode]
        test_series = test_data["series"][mode]

        # Compare bars
        compare_bars(golden_series["bars"], test_series["bars"], mode, result)

        if not [e for e in result.errors if mode in e]:
            print(f"  ✅ {mode}: All {len(golden_series['bars'])} bars match")

    # Print report
    return result.report()


def validate_key_points(test_data: dict, golden_path: Path):
    """
    Quick validation of key data points.

    This is a fast sanity check that can be run during development.
    """
    golden_data = load_golden_output(golden_path)

    print("\n" + "=" * 100)
    print("KEY VALIDATION POINTS")
    print("=" * 100)

    key_dates = ["2020-08-07", "2020-08-28", "2020-08-31", "2020-09-02"]
    all_match = True

    for date in key_dates:
        print(f"\n{date}:")
        for mode in ["unadjusted", "adjusted", "total_return"]:
            golden_bar = next((b for b in golden_data["series"][mode]["bars"] if b["trade_datetime"] == date), None)
            test_bar = next((b for b in test_data["series"][mode]["bars"] if b["trade_datetime"] == date), None)

            if golden_bar and test_bar:
                close_match = abs(golden_bar["close"] - test_bar["close"]) < 0.01
                vol_match = golden_bar["volume"] == test_bar["volume"]
                match_str = "✅" if (close_match and vol_match) else "❌"

                if not (close_match and vol_match):
                    all_match = False

                print(
                    f"  {match_str} {mode:15} | C: ${test_bar['close']:8.2f} (exp: ${golden_bar['close']:8.2f}) "
                    f"V: {test_bar['volume']:12,} (exp: {golden_bar['volume']:,})"
                )

    if all_match:
        print("\n✅ All key validation points match!")
    else:
        print("\n❌ Some key validation points don't match!")

    return all_match


if __name__ == "__main__":
    print("=" * 100)
    print("GOLDEN OUTPUT VALIDATION TEST")
    print("=" * 100)
    print("\nThis test will be used after refactoring to validate correctness.")
    print("It compares the output of the refactored implementation against the golden output.")

    golden_path = Path("src/qtrader/1playground/golden_output.json")

    if not golden_path.exists():
        print(f"\n❌ Golden output not found: {golden_path}")
        print("Run test_golden_output.py first to generate it.")
    else:
        print(f"\n✅ Golden output found: {golden_path}")
        golden_data = load_golden_output(golden_path)
        print(
            f"✅ Contains {len(golden_data['series'])} series with {golden_data['series']['unadjusted']['bar_count']} bars each"
        )
        print("\nAfter refactoring, call validate_against_golden(test_data, golden_path) to validate.")
