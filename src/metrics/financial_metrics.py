"""Financial metrics for evaluating trading strategies."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Final[tuple[str, str, str]] = (
    "portfolio_value",
    "strategy_return",
    "position",
)
EPSILON: Final[float] = 1e-12


def _validate_metrics_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the input DataFrame and return a clean numeric copy."""

    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

    frame = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows remain after cleaning metric input data.")

    return frame


def _coerce_returns_array(returns: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """Convert periodic returns into a clean NumPy array."""

    returns_array = np.asarray(returns, dtype=float)
    returns_array = returns_array[np.isfinite(returns_array)]
    return returns_array


def _growth_factor(returns: np.ndarray) -> float:
    """Calculate compounded growth factor from a return series."""

    if returns.size == 0:
        return 1.0

    return float(np.prod(1.0 + returns))


def _annualized_return_from_returns(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized return directly from periodic returns."""

    if returns.size == 0 or periods_per_year <= 0:
        return 0.0

    growth_factor = _growth_factor(returns)
    if growth_factor <= 0.0:
        return -1.0

    number_of_years = returns.size / periods_per_year
    if number_of_years <= 0.0:
        return 0.0

    return float(growth_factor ** (1.0 / number_of_years) - 1.0)


def total_return(df: pd.DataFrame) -> float:
    """Calculate the total compounded return of the strategy."""

    frame = _validate_metrics_frame(df)
    returns = frame["strategy_return"].to_numpy(dtype=float)
    return float(_growth_factor(returns) - 1.0)


def annualized_return(
    df: pd.DataFrame,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized return of the strategy."""

    frame = _validate_metrics_frame(df)
    returns = frame["strategy_return"].to_numpy(dtype=float)
    return _annualized_return_from_returns(
        returns=returns,
        periods_per_year=periods_per_year,
    )


def annualized_volatility(
    df: pd.DataFrame,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility from periodic strategy returns."""

    frame = _validate_metrics_frame(df)
    returns = frame["strategy_return"].to_numpy(dtype=float)

    if returns.size < 2 or periods_per_year <= 0:
        return 0.0

    volatility = np.std(returns, ddof=1)
    return float(volatility * np.sqrt(periods_per_year))


def sharpe_ratio(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio."""

    frame = _validate_metrics_frame(df)
    returns = frame["strategy_return"].to_numpy(dtype=float)

    if returns.size < 2 or periods_per_year <= 0:
        return 0.0

    periodic_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - periodic_risk_free_rate
    excess_volatility = np.std(excess_returns, ddof=1)

    if np.isclose(excess_volatility, 0.0, atol=EPSILON):
        return 0.0

    return float(np.mean(excess_returns) / excess_volatility * np.sqrt(periods_per_year))


def max_drawdown(df: pd.DataFrame) -> float:
    """Calculate the maximum drawdown based on portfolio value."""

    frame = _validate_metrics_frame(df)
    portfolio_values = frame["portfolio_value"].to_numpy(dtype=float)

    if portfolio_values.size == 0:
        return 0.0

    running_max = np.maximum.accumulate(portfolio_values)
    valid_denominator = np.where(running_max > EPSILON, running_max, np.nan)
    drawdowns = portfolio_values / valid_denominator - 1.0
    drawdowns = np.where(np.isfinite(drawdowns), drawdowns, 0.0)

    return float(np.min(drawdowns))


def win_rate(df: pd.DataFrame) -> float:
    """Calculate the fraction of profitable non-zero return periods."""

    frame = _validate_metrics_frame(df)
    returns = frame["strategy_return"].to_numpy(dtype=float)
    active_returns = returns[np.abs(returns) > EPSILON]

    if active_returns.size == 0:
        return 0.0

    return float(np.mean(active_returns > 0.0))


def number_of_trades(df: pd.DataFrame) -> int:
    """Count the number of position changes, including the initial entry."""

    frame = _validate_metrics_frame(df)
    positions = frame["position"].to_numpy(dtype=float)

    if positions.size == 0:
        return 0

    changes = np.abs(np.diff(positions, prepend=positions[0]))
    changes[0] = abs(positions[0])
    return int(np.sum(changes > EPSILON))


def calmar_ratio(
    df: pd.DataFrame,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Calmar ratio as annualized return divided by max drawdown."""

    annual_return = annualized_return(df=df, periods_per_year=periods_per_year)
    drawdown = abs(max_drawdown(df=df))

    if np.isclose(drawdown, 0.0, atol=EPSILON):
        return 0.0

    return float(annual_return / drawdown)


def calculate_all_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Calculate the full set of trading strategy performance metrics."""

    return {
        "total_return": total_return(df=df),
        "annualized_return": annualized_return(
            df=df,
            periods_per_year=periods_per_year,
        ),
        "annualized_volatility": annualized_volatility(
            df=df,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(
            df=df,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "max_drawdown": max_drawdown(df=df),
        "win_rate": win_rate(df=df),
        "number_of_trades": number_of_trades(df=df),
        "calmar_ratio": calmar_ratio(
            df=df,
            periods_per_year=periods_per_year,
        ),
    }


def calculate_financial_metrics(
    df: pd.DataFrame | None = None,
    *,
    returns: np.ndarray | pd.Series | list[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Backward-compatible wrapper for project modules.

    Args:
        df: Strategy results DataFrame with ``portfolio_value``,
            ``strategy_return``, and ``position`` columns.
        returns: Optional array-like of periodic returns for legacy usage.
        risk_free_rate: Annual risk-free rate used in the Sharpe ratio.
        periods_per_year: Number of return periods per trading year.

    Returns:
        Dictionary with performance metrics. If only ``returns`` are provided,
        trade-dependent metrics are returned as ``0.0``.
    """

    if df is not None:
        return calculate_all_metrics(
            df=df,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    if returns is None:
        raise ValueError("Either df or returns must be provided.")

    returns_array = _coerce_returns_array(returns)
    volatility = 0.0
    sharpe = 0.0

    if returns_array.size >= 2 and periods_per_year > 0:
        volatility = float(np.std(returns_array, ddof=1) * np.sqrt(periods_per_year))
        periodic_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = returns_array - periodic_risk_free_rate
        excess_volatility = np.std(excess_returns, ddof=1)
        if not np.isclose(excess_volatility, 0.0, atol=EPSILON):
            sharpe = float(
                np.mean(excess_returns) / excess_volatility * np.sqrt(periods_per_year)
            )

    return {
        "total_return": float(_growth_factor(returns_array) - 1.0),
        "annualized_return": _annualized_return_from_returns(
            returns=returns_array,
            periods_per_year=periods_per_year,
        ),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": 0.0,
        "win_rate": float(np.mean(returns_array > 0.0)) if returns_array.size > 0 else 0.0,
        "number_of_trades": 0.0,
        "calmar_ratio": 0.0,
    }
