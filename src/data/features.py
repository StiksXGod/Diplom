"""Feature engineering utilities for market OHLCV time series."""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

DEFAULT_FEATURE_COLUMNS: Final[list[str]] = [
    "simple_return",
    "log_return",
    "rolling_mean_5",
    "rolling_mean_10",
    "rolling_mean_20",
    "rolling_std_10",
    "rolling_std_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "bollinger_upper",
    "bollinger_lower",
]

TARGET_COLUMNS: Final[list[str]] = [
    "target_direction",
    "target_return",
]


def _validate_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the input schema and return a normalized copy."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

    frame = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "Date"]
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if frame.empty:
        raise ValueError("Input DataFrame must not be empty.")

    return frame


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index using Wilder smoothing."""

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(average_loss != 0.0, 100.0)
    rsi = rsi.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)

    return rsi


def _calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Calculate MACD and the corresponding signal line."""

    fast_ema = close.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(
        span=signal_period,
        adjust=False,
        min_periods=signal_period,
    ).mean()
    return macd, macd_signal


def build_feature_set(data: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features and targets from OHLCV market data.

    Args:
        data: Input OHLCV DataFrame with columns ``Date``, ``Open``, ``High``,
            ``Low``, ``Close``, ``Adj Close``, and ``Volume``.

    Returns:
        DataFrame with original columns, engineered features, and target
        variables. Rows containing ``NaN`` values after feature generation are
        removed.
    """

    features = _validate_input_frame(data)
    close = features["Close"].astype(float)

    features["simple_return"] = close.pct_change()
    features["log_return"] = np.log(close / close.shift(1))

    features["rolling_mean_5"] = close.rolling(window=5).mean()
    features["rolling_mean_10"] = close.rolling(window=10).mean()
    features["rolling_mean_20"] = close.rolling(window=20).mean()

    features["rolling_std_10"] = close.rolling(window=10).std()
    features["rolling_std_20"] = close.rolling(window=20).std()

    features["rsi_14"] = _calculate_rsi(close=close, period=14)

    macd, macd_signal = _calculate_macd(close=close)
    features["macd"] = macd
    features["macd_signal"] = macd_signal

    features["bollinger_upper"] = features["rolling_mean_20"] + 2.0 * features["rolling_std_20"]
    features["bollinger_lower"] = features["rolling_mean_20"] - 2.0 * features["rolling_std_20"]

    next_close = close.shift(-1)
    features["target_direction"] = (next_close > close).astype(int)
    features["target_return"] = next_close / close - 1.0

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna().reset_index(drop=True)


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Alias for :func:`build_feature_set` for explicit pipeline naming."""

    return build_feature_set(data=data)


def select_feature_columns(
    feature_frame: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return a subset of engineered feature columns."""

    missing_columns = [column for column in columns if column not in feature_frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Missing required feature columns: {missing}.")

    return feature_frame.loc[:, list(columns)].copy()


def split_time_series(
    df: pd.DataFrame,
    train_size: int | float,
    val_size: int | float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data into train, validation, and test parts.

    Args:
        df: Input DataFrame sorted in chronological order.
        train_size: Training size as an absolute number of rows or a fraction in
            the ``(0, 1)`` interval.
        val_size: Validation size as an absolute number of rows or a fraction in
            the ``(0, 1)`` interval.

    Returns:
        Tuple of ``(train_df, val_df, test_df)`` without any shuffling.
    """

    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    total_rows = len(df)
    train_rows = _resolve_split_size(size=train_size, total_rows=total_rows, name="train_size")
    val_rows = _resolve_split_size(size=val_size, total_rows=total_rows, name="val_size")

    if train_rows + val_rows >= total_rows:
        raise ValueError(
            "train_size and val_size leave no rows for the test split. "
            "Reduce one of the split sizes."
        )

    train_df = df.iloc[:train_rows].copy()
    val_df = df.iloc[train_rows : train_rows + val_rows].copy()
    test_df = df.iloc[train_rows + val_rows :].copy()

    return train_df, val_df, test_df


def _resolve_split_size(size: int | float, total_rows: int, name: str) -> int:
    """Convert a split size from rows or fraction into an integer count."""

    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} must be in the (0, 1) interval when passed as float.")
        rows = int(total_rows * size)
    elif isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        rows = size
    else:
        raise TypeError(f"{name} must be either int or float.")

    if rows <= 0:
        raise ValueError(f"{name} results in an empty split.")

    return rows
