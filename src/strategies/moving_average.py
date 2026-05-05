"""Moving-average crossover benchmark strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


REQUIRED_COLUMNS: Final[tuple[str, str]] = ("Date", "Close")


def _validate_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and return a normalized copy."""

    if data.empty:
        raise ValueError("Input DataFrame must not be empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

    frame = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid rows remain after cleaning Date and Close columns.")
    if (frame["Close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")

    return frame


@dataclass(slots=True)
class MovingAverageCrossoverStrategy:
    """Trade long when the short moving average is above the long moving average."""

    short_window: int = 20
    long_window: int = 50
    initial_cash: float = 10_000.0
    transaction_cost: float = 0.001

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the moving-average crossover backtest.

        Args:
            data: Input market data with at least ``Date`` and ``Close`` columns.

        Returns:
            DataFrame with daily portfolio state and returns.
        """

        self._validate_parameters()
        frame = _validate_market_data(data)

        if len(frame) < self.long_window:
            raise ValueError(
                "Input DataFrame must contain at least long_window rows "
                "to compute the moving-average signals."
            )

        short_ma = frame["Close"].rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ma = frame["Close"].rolling(window=self.long_window, min_periods=self.long_window).mean()

        frame["position"] = (short_ma > long_ma).astype(float)
        frame["daily_return"] = frame["Close"].pct_change().fillna(0.0)

        exposure = frame["position"].shift(1).fillna(0.0)
        turnover = frame["position"].diff().abs().fillna(frame["position"])
        frame["strategy_return"] = exposure * frame["daily_return"] - (
            turnover * self.transaction_cost
        )
        frame["portfolio_value"] = self.initial_cash * (1.0 + frame["strategy_return"]).cumprod()

        return frame[
            ["Date", "Close", "position", "daily_return", "strategy_return", "portfolio_value"]
        ].copy()

    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""

        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("short_window and long_window must be positive integers.")
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if not 0.0 <= self.transaction_cost < 1.0:
            raise ValueError("transaction_cost must be in the [0, 1) interval.")
