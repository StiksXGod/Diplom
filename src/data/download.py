"""Utilities for downloading historical OHLCV market data via yfinance."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS: Final[list[str]] = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be downloaded from the remote source."""


class EmptyDataError(ValueError):
    """Raised when a download request succeeds but returns no rows."""


class InvalidTickerError(ValueError):
    """Raised when a ticker symbol is invalid or unavailable."""


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize the input ticker."""

    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker:
        raise InvalidTickerError("Ticker must not be empty.")
    if " " in normalized_ticker:
        raise InvalidTickerError("Ticker must not contain spaces.")

    return normalized_ticker


def _validate_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """Validate the requested date range and return normalized ISO dates."""

    try:
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)
    except Exception as error:
        raise ValueError("start_date and end_date must be valid dates.") from error

    if start_timestamp >= end_timestamp:
        raise ValueError("start_date must be earlier than end_date.")

    return start_timestamp.strftime("%Y-%m-%d"), end_timestamp.strftime("%Y-%m-%d")


def _flatten_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert potential yfinance MultiIndex columns into plain column names."""

    if not isinstance(data.columns, pd.MultiIndex):
        return data

    flattened_columns: list[str] = []
    ticker_upper = ticker.upper()

    for column in data.columns.to_flat_index():
        if not isinstance(column, tuple):
            flattened_columns.append(str(column))
            continue

        parts = [str(part) for part in column if str(part).upper() != ticker_upper]
        flattened_columns.append(parts[0] if parts else str(column[0]))

    normalized = data.copy()
    normalized.columns = flattened_columns
    return normalized


def _ensure_valid_remote_ticker(ticker: str) -> None:
    """Check whether the ticker exists on Yahoo Finance."""

    try:
        probe_data = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
    except Exception as error:
        raise DataDownloadError(
            f"Failed to validate ticker '{ticker}' after an empty download."
        ) from error

    if probe_data.empty:
        raise InvalidTickerError(
            f"Ticker '{ticker}' is invalid or unavailable in Yahoo Finance."
        )


def _prepare_output_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize raw yfinance output to the required OHLCV schema."""

    normalized = _flatten_columns(data=data, ticker=ticker).reset_index()
    if "Datetime" in normalized.columns and "Date" not in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Date"})

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_columns:
        missing_columns_str = ", ".join(missing_columns)
        raise DataDownloadError(
            "Downloaded dataset has an unexpected schema. "
            f"Missing columns: {missing_columns_str}."
        )

    output = normalized.loc[:, REQUIRED_COLUMNS].copy()
    output["Date"] = pd.to_datetime(output["Date"])
    return output


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Download historical OHLCV quotes from Yahoo Finance.

    Args:
        ticker: Exchange ticker symbol, for example ``AAPL``.
        start_date: Inclusive start date in ISO-like string format.
        end_date: Exclusive end date in ISO-like string format.
        save_path: Optional output path for saving the downloaded dataset as CSV.

    Returns:
        A pandas DataFrame with columns ``Date``, ``Open``, ``High``, ``Low``,
        ``Close``, ``Adj Close``, and ``Volume``.

    Raises:
        InvalidTickerError: If the ticker is malformed or unavailable.
        EmptyDataError: If the selected period contains no data.
        DataDownloadError: If the remote download fails.
        ValueError: If the date range is invalid.
    """

    normalized_ticker = _validate_ticker(ticker)
    normalized_start_date, normalized_end_date = _validate_date_range(
        start_date=start_date,
        end_date=end_date,
    )

    try:
        raw_data = yf.download(
            tickers=normalized_ticker,
            start=normalized_start_date,
            end=normalized_end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception as error:
        raise DataDownloadError(
            f"Failed to download data for ticker '{normalized_ticker}'."
        ) from error

    if raw_data.empty:
        _ensure_valid_remote_ticker(normalized_ticker)
        raise EmptyDataError(
            "Downloaded DataFrame is empty. "
            f"No quotes found for '{normalized_ticker}' in the selected period."
        )

    dataset = _prepare_output_frame(data=raw_data, ticker=normalized_ticker)
    if dataset.empty:
        raise EmptyDataError(
            f"Downloaded DataFrame for ticker '{normalized_ticker}' is empty after processing."
        )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

    return dataset


if __name__ == "__main__":
    SAMPLE_TICKER = "AAPL"
    SAMPLE_START_DATE = "2023-01-01"
    SAMPLE_END_DATE = "2023-12-31"
    SAMPLE_SAVE_PATH = "data/raw/aapl_2023.csv"

    try:
        ohlcv_data = download_ohlcv(
            ticker=SAMPLE_TICKER,
            start_date=SAMPLE_START_DATE,
            end_date=SAMPLE_END_DATE,
            save_path=SAMPLE_SAVE_PATH,
        )
        print(ohlcv_data.head())
        print(f"Downloaded {len(ohlcv_data)} rows to '{SAMPLE_SAVE_PATH}'.")
    except (InvalidTickerError, EmptyDataError, DataDownloadError, ValueError) as error:
        print(f"Download error: {error}")
