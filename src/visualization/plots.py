"""Visualization helpers for trading strategies and model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def _normalize_save_path(save_path: str | Path | None) -> Path | None:
    """Normalize a save path and enforce the PNG extension."""

    if save_path is None:
        return None

    path = Path(save_path)
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(figure: Figure, save_path: str | Path | None) -> None:
    """Save a figure to disk if a path is provided."""

    normalized_path = _normalize_save_path(save_path)
    if normalized_path is not None:
        figure.savefig(normalized_path, dpi=150, bbox_inches="tight")


def _validate_date_close_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a frame containing at least ``Date`` and ``Close`` columns."""

    if data.empty:
        raise ValueError("Input DataFrame must not be empty.")

    required_columns = ["Date", "Close"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid rows remain after cleaning Date and Close columns.")

    return frame


def _validate_portfolio_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a frame containing portfolio value history."""

    if data.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if "portfolio_value" not in data.columns:
        raise KeyError("Input DataFrame must contain a 'portfolio_value' column.")

    frame = data.copy()
    frame["portfolio_value"] = pd.to_numeric(frame["portfolio_value"], errors="coerce")

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "portfolio_value"]).sort_values("Date")
    else:
        frame = frame.dropna(subset=["portfolio_value"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("No valid rows remain after cleaning portfolio data.")

    return frame.reset_index(drop=True)


def _infer_signal_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Infer buy and sell signal masks from either action or position columns."""

    if "action" in frame.columns:
        actions = pd.to_numeric(frame["action"], errors="coerce").fillna(0).astype(int)
        return actions.eq(1), actions.eq(2)

    if "position" in frame.columns:
        positions = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
        changes = positions.diff().fillna(positions)
        return changes > 0, changes < 0

    raise KeyError("Input DataFrame must contain either 'action' or 'position'.")


def plot_price_with_signals(
    strategy_results: pd.DataFrame,
    title: str = "Price With Trading Signals",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the close price with buy and sell markers.

    Args:
        strategy_results: DataFrame with at least ``Date`` and ``Close`` plus
            either ``action`` or ``position``.
        title: Plot title.
        save_path: Optional file path for saving the figure as PNG.

    Returns:
        A matplotlib Figure.
    """

    frame = _validate_date_close_frame(strategy_results)
    buy_mask, sell_mask = _infer_signal_masks(frame)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(frame["Date"], frame["Close"], label="Close", color="tab:blue", linewidth=1.6)

    if buy_mask.any():
        axis.scatter(
            frame.loc[buy_mask, "Date"],
            frame.loc[buy_mask, "Close"],
            marker="^",
            color="tab:green",
            s=70,
            label="Buy",
            zorder=3,
        )
    if sell_mask.any():
        axis.scatter(
            frame.loc[sell_mask, "Date"],
            frame.loc[sell_mask, "Close"],
            marker="v",
            color="tab:red",
            s=70,
            label="Sell",
            zorder=3,
        )

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()

    _save_figure(figure, save_path)
    return figure


def plot_equity_curves(
    strategy_results: Mapping[str, pd.DataFrame],
    title: str = "Equity Curve Comparison",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot portfolio value curves for multiple strategies.

    Args:
        strategy_results: Mapping from strategy name to a DataFrame with a
            ``portfolio_value`` column and optional ``Date`` column.
        title: Plot title.
        save_path: Optional file path for saving the figure as PNG.

    Returns:
        A matplotlib Figure.
    """

    if not strategy_results:
        raise ValueError("strategy_results must not be empty.")

    figure, axis = plt.subplots(figsize=(14, 6))

    for strategy_name, result_frame in strategy_results.items():
        frame = _validate_portfolio_frame(result_frame)
        x_values: Sequence[Any]
        if "Date" in frame.columns:
            x_values = frame["Date"]
        else:
            x_values = frame.index

        axis.plot(
            x_values,
            frame["portfolio_value"],
            linewidth=1.8,
            label=strategy_name,
        )

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()

    _save_figure(figure, save_path)
    return figure


def plot_drawdown(
    strategy_results: pd.DataFrame,
    title: str = "Portfolio Drawdown",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot the drawdown curve for a single strategy.

    Args:
        strategy_results: DataFrame with ``portfolio_value`` and optional
            ``Date`` columns.
        title: Plot title.
        save_path: Optional file path for saving the figure as PNG.

    Returns:
        A matplotlib Figure.
    """

    frame = _validate_portfolio_frame(strategy_results)
    portfolio_values = frame["portfolio_value"].astype(float)
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1.0

    if "Date" in frame.columns:
        x_values: Sequence[Any] = frame["Date"]
    else:
        x_values = frame.index

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(x_values, drawdown, color="tab:red", linewidth=1.6, label="Drawdown")
    axis.fill_between(x_values, drawdown, 0.0, color="tab:red", alpha=0.2)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()

    _save_figure(figure, save_path)
    return figure


def plot_metrics_bar(
    metrics_data: pd.DataFrame | Mapping[str, Mapping[str, float | int]],
    metric_name: str,
    label_column: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Build a bar chart for one metric across multiple strategies.

    Args:
        metrics_data: DataFrame or mapping with per-strategy metrics.
        metric_name: Metric column to visualize.
        label_column: Optional column containing strategy names.
        title: Optional plot title.
        save_path: Optional file path for saving the figure as PNG.

    Returns:
        A matplotlib Figure.
    """

    if isinstance(metrics_data, pd.DataFrame):
        frame = metrics_data.copy()
    else:
        frame = pd.DataFrame.from_dict(metrics_data, orient="index").reset_index()
        frame = frame.rename(columns={"index": "strategy"})

    if frame.empty:
        raise ValueError("metrics_data must not be empty.")
    if metric_name not in frame.columns:
        raise KeyError(f"Metric '{metric_name}' is not present in metrics_data.")

    if label_column is None:
        for candidate in ("strategy", "strategy_name", "model_name", "name"):
            if candidate in frame.columns:
                label_column = candidate
                break

    if label_column is None:
        label_series = frame.index.astype(str)
    else:
        if label_column not in frame.columns:
            raise KeyError(f"Label column '{label_column}' is not present in metrics_data.")
        label_series = frame[label_column].astype(str)

    values = pd.to_numeric(frame[metric_name], errors="coerce")
    valid_mask = values.notna()
    labels = pd.Index(label_series)[valid_mask]
    metric_values = values[valid_mask]

    if metric_values.empty:
        raise ValueError(f"Metric '{metric_name}' has no valid numeric values.")

    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(labels, metric_values, color="tab:blue", alpha=0.85)
    axis.set_title(title or f"{metric_name} Comparison")
    axis.set_xlabel("Strategy")
    axis.set_ylabel(metric_name)
    axis.grid(axis="y", alpha=0.3)
    axis.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, metric_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    _save_figure(figure, save_path)
    return figure


def plot_price_series(
    quotes: pd.DataFrame,
    price_column: str = "Close",
    title: str = "Asset Price History",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a single price series.

    This helper is kept for compatibility with earlier versions of the project.
    """

    if quotes.empty:
        raise ValueError("quotes must not be empty.")
    if price_column not in quotes.columns:
        raise KeyError(f"Column '{price_column}' is not present in the dataset.")

    frame = quotes.copy()
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
        frame = frame.dropna(subset=["Date", price_column]).sort_values("Date")
        x_values: Sequence[Any] = frame["Date"]
    else:
        frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce")
        frame = frame.dropna(subset=[price_column]).reset_index(drop=True)
        x_values = frame.index

    if frame.empty:
        raise ValueError("No valid rows remain after cleaning the price series.")

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(x_values, frame[price_column], label=price_column, linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()

    _save_figure(figure, save_path)
    return figure


def plot_equity_curve(
    equity_curve: Sequence[float],
    title: str = "Strategy Equity Curve",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot one equity curve.

    This helper is kept for compatibility with earlier versions of the project.
    """

    values = [float(value) for value in equity_curve]
    if not values:
        raise ValueError("equity_curve must not be empty.")

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(values, linewidth=1.5, color="tab:green")
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel("Portfolio value")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    _save_figure(figure, save_path)
    return figure
