"""
Multi-mode bar model - container for all adjustment modes.

This module defines the MultiBar model which contains all three adjustment
modes (unadjusted, adjusted, total_return) in a single container. This allows
different components to select the optimal mode for their purpose:

- Strategy: adjusted (split-consistent indicators)
- Execution: unadjusted (realistic fills at actual prices)
- Performance: total_return (includes dividend reinvestment)
"""

import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qtrader.models.bar import Bar

AdjustmentMode = Literal["unadjusted", "adjusted", "total_return"]


class MultiBar(BaseModel):
    """
    Bar containing all adjustment modes.

    This container provides access to the same bar data in all three adjustment
    modes simultaneously. Each component can select the mode that best suits its
    purpose without requiring separate data loads.

    Attributes:
        symbol: Ticker symbol
        trade_datetime: Trading datetime (accepts datetime or ISO string)
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
    trade_datetime: Any = Field(..., description="Trade datetime (accepts datetime or ISO string)")
    unadjusted: Bar = Field(..., description="Unadjusted prices (actual traded)")
    adjusted: Bar = Field(..., description="Split-adjusted prices")
    total_return: Bar = Field(..., description="Split + dividend adjusted")

    model_config = ConfigDict(frozen=True)  # Immutable

    @field_validator("trade_datetime", mode="before")
    @classmethod
    def parse_trade_datetime(cls, v: Any) -> datetime.datetime:
        """
        Parse trade_datetime from datetime or ISO string.

        Args:
            v: datetime object or ISO format string

        Returns:
            datetime.datetime object

        Raises:
            ValueError: If input cannot be parsed
        """
        if isinstance(v, datetime.datetime):
            return v
        elif isinstance(v, str):
            # Parse ISO format string
            return datetime.datetime.fromisoformat(v)
        else:
            raise ValueError(f"Cannot parse datetime from type {type(v)}: {v}")

    def get_bar(self, mode: AdjustmentMode) -> Bar:
        """
        Get bar for specific adjustment mode.

        Args:
            mode: Adjustment mode ('unadjusted', 'adjusted', 'total_return')

        Returns:
            Bar for requested mode

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
