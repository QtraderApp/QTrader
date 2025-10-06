"""Dividend calculator for corporate actions."""

from decimal import Decimal
from typing import Optional

from qtrader.config.logging_config import LoggerFactory

logger = LoggerFactory.get_logger()


class DividendCalculator:
    """
    Calculate dividend per share from adjustment factors.

    Used to derive dividend amounts from vendor adjustment metadata
    (specifically Algoseek cumulative price factors).
    """

    @staticmethod
    def calculate_from_factors(
        close_before: Decimal,
        close_after: Decimal,
        cumulative_price_factor: Decimal,
    ) -> Optional[Decimal]:
        """
        Calculate dividend per share from adjustment factors.

        Formula:
            div = close_after * (cumulative_price_factor - 1)

        This formula works because:
        - cumulative_price_factor = close_before / close_after
        - When a dividend is paid, close_after drops by the dividend amount
        - factor - 1 gives us the fractional change
        - Multiplying by close_after recovers the dividend amount

        Args:
            close_before: Close price day before ex-date (adjusted)
            close_after: Close price on ex-date (adjusted)
            cumulative_price_factor: Cumulative adjustment factor from vendor
                                      (ratio of close_before / close_after)

        Returns:
            Dividend per share (Decimal), or None if invalid

        Example:
            AAPL 2023-02-10 dividend:
            close_before = $152.55 (adjusted)
            close_after = $152.32 (adjusted)
            cumulative_price_factor = 1.001508 (152.55/152.32)
            div = 152.32 * (1.001508 - 1) = 152.32 * 0.001508 = $0.23/share
        """
        # Validation
        if close_before <= Decimal("0"):
            logger.warning(
                "dividend_calculator.invalid_price_before",
                close_before=float(close_before),
            )
            return None

        if close_after <= Decimal("0"):
            logger.warning(
                "dividend_calculator.invalid_price_after",
                close_after=float(close_after),
            )
            return None

        if cumulative_price_factor <= Decimal("0"):
            logger.warning(
                "dividend_calculator.invalid_factor",
                cumulative_price_factor=float(cumulative_price_factor),
            )
            return None

        # Calculate dividend
        try:
            # factor - 1 gives the fractional dividend yield
            # Multiply by close_after to get the dividend amount
            dividend_per_share = close_after * (cumulative_price_factor - Decimal("1"))

            # Dividend should be positive (or very small negative due to rounding)
            if dividend_per_share < Decimal("-0.01"):
                logger.warning(
                    "dividend_calculator.negative_dividend",
                    close_before=float(close_before),
                    close_after=float(close_after),
                    factor=float(cumulative_price_factor),
                    calculated_div=float(dividend_per_share),
                )
                return None

            # Round to nearest cent
            dividend_per_share = dividend_per_share.quantize(Decimal("0.01"))

            logger.debug(
                "dividend_calculator.calculated",
                close_before=float(close_before),
                close_after=float(close_after),
                factor=float(cumulative_price_factor),
                dividend=float(dividend_per_share),
            )

            return dividend_per_share if dividend_per_share > Decimal("0") else None

        except Exception as e:
            logger.error(
                "dividend_calculator.calculation_error",
                error=str(e),
                close_before=float(close_before),
                close_after=float(close_after),
                factor=float(cumulative_price_factor),
            )
            return None

    @staticmethod
    def validate_dividend_event(
        symbol: str,
        ex_date: str,
        dividend_per_share: Decimal,
    ) -> bool:
        """
        Validate dividend event parameters.

        Args:
            symbol: Trading symbol
            ex_date: Ex-dividend date (ISO format)
            dividend_per_share: Calculated dividend amount

        Returns:
            True if valid, False otherwise
        """
        if not symbol:
            logger.warning("dividend_calculator.missing_symbol")
            return False

        if not ex_date:
            logger.warning("dividend_calculator.missing_ex_date", symbol=symbol)
            return False

        if dividend_per_share <= Decimal("0"):
            logger.warning(
                "dividend_calculator.invalid_amount",
                symbol=symbol,
                ex_date=ex_date,
                dividend=float(dividend_per_share),
            )
            return False

        # Sanity check: dividend shouldn't be more than $100/share
        if dividend_per_share > Decimal("100.00"):
            logger.warning(
                "dividend_calculator.suspiciously_high",
                symbol=symbol,
                ex_date=ex_date,
                dividend=float(dividend_per_share),
            )
            return False

        return True
