"""
Multi-mode bar model - container for all adjustment modes.

This module defines the MultiModeBar model which contains all three adjustment
modes (unadjusted, adjusted, total_return) in a single container. This allows
different components to select the optimal mode for their purpose:

- Strategy: adjusted (split-consistent indicators)
- Execution: unadjusted (realistic fills at actual prices)
- Performance: total_return (includes dividend reinvestment)
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qtrader.models.canonical_bar import CanonicalBar

AdjustmentMode = Literal["unadjusted", "adjusted", "total_return"]


class MultiModeBar(BaseModel):
    """
    Bar containing all adjustment modes.

    This container provides access to the same bar data in all three adjustment
    modes simultaneously. Each component can select the mode that best suits its
    purpose without requiring separate data loads.

    Attributes:
        symbol: Ticker symbol
        trade_datetime: Trading datetime (ISO format)
        unadjusted: Raw prices as traded (for execution/fills)
        adjusted: Split-adjusted prices (for indicators/signals)
        total_return: Split + dividend adjusted (for performance)

    Examples:
        >>> # Strategy uses adjusted for indicators
        >>> strategy_bar = multi_bar.adjusted
        >>> sma = calculate_sma(strategy_bar.close)
        >>>
        >>> # Execution uses unadjusted for fills
        >>> exec_bar = multi_bar.unadjusted
        >>> fill_price = exec_bar.high
        >>> commission = fill_price * shares * 0.001
        >>>
        >>> # Performance uses total_return
        >>> perf_bar = multi_bar.total_return
        >>> return_pct = (perf_bar.close - entry) / entry

    Notes:
        - All three bars share the same trade_datetime
        - Immutable after creation (frozen=True)
        - Use get_bar() for dynamic mode selection
    """

    symbol: str = Field(..., description="Ticker symbol")
    trade_datetime: str = Field(..., description="Trade datetime (ISO format)")
    unadjusted: CanonicalBar = Field(..., description="Unadjusted prices (actual traded)")
    adjusted: CanonicalBar = Field(..., description="Split-adjusted prices")
    total_return: CanonicalBar = Field(..., description="Split + dividend adjusted")

    model_config = ConfigDict(frozen=True)  # Immutable

    def get_bar(self, mode: AdjustmentMode) -> CanonicalBar:
        """
        Get bar for specific adjustment mode.

        Args:
            mode: Adjustment mode ('unadjusted', 'adjusted', 'total_return')

        Returns:
            CanonicalBar for requested mode

        Raises:
            ValueError: If mode is invalid

        Examples:
            >>> bar = multi_bar.get_bar("adjusted")
            >>> bar = multi_bar.get_bar(config.signal_generation_mode)
        """
        if mode == "unadjusted":
            return self.unadjusted
        elif mode == "adjusted":
            return self.adjusted
        elif mode == "total_return":
            return self.total_return
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'unadjusted', 'adjusted', or 'total_return'")
