"""Run the full experiment pipeline for comparing trading approaches."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.download import download_ohlcv
from src.data.features import DEFAULT_FEATURE_COLUMNS, build_feature_set, split_time_series
from src.metrics.financial_metrics import calculate_all_metrics
from src.models.ml_models import train_and_evaluate_models
from src.strategies.buy_hold import BuyAndHoldStrategy
from src.strategies.moving_average import MovingAverageCrossoverStrategy
from src.visualization.plots import (
    plot_drawdown,
    plot_equity_curves,
    plot_metrics_bar,
    plot_price_with_signals,
)


@dataclass(slots=True)
class ReportPaths:
    """Directory layout for saved experiment artifacts."""

    run_dir: Path
    data_dir: Path
    metrics_dir: Path
    plots_dir: Path
    predictions_dir: Path
    strategies_dir: Path
    agents_dir: Path
    equity_dir: Path

    @classmethod
    def create(cls, reports_root: Path, run_name: str) -> "ReportPaths":
        """Create and return the directory structure for one experiment run."""

        run_dir = reports_root / run_name
        paths = cls(
            run_dir=run_dir,
            data_dir=run_dir / "data",
            metrics_dir=run_dir / "metrics",
            plots_dir=run_dir / "plots",
            predictions_dir=run_dir / "predictions",
            strategies_dir=run_dir / "strategies",
            agents_dir=run_dir / "agents",
            equity_dir=run_dir / "equity_curves",
        )
        for path in (
            paths.run_dir,
            paths.data_dir,
            paths.metrics_dir,
            paths.plots_dir,
            paths.predictions_dir,
            paths.strategies_dir,
            paths.agents_dir,
            paths.equity_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full experiment pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the complete diploma experiment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol for Yahoo Finance.")
    parser.add_argument("--start_date", required=True, help="Experiment start date in YYYY-MM-DD.")
    parser.add_argument("--end_date", required=True, help="Experiment end date in YYYY-MM-DD.")
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=10_000.0,
        help="Initial capital for all backtests.",
    )
    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.001,
        help="Transaction cost applied to trades.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="Training split size as a fraction of the feature dataset.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation split size as a fraction of the feature dataset.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--short_window",
        type=int,
        default=20,
        help="Short moving-average window for the benchmark crossover strategy.",
    )
    parser.add_argument(
        "--long_window",
        type=int,
        default=50,
        help="Long moving-average window for the benchmark crossover strategy.",
    )
    parser.add_argument(
        "--rl_window_size",
        type=int,
        default=30,
        help="Observation window size used by the RL environment.",
    )
    parser.add_argument(
        "--reports_dir",
        default="reports",
        help="Root directory for saved experiment outputs.",
    )
    parser.add_argument(
        "--rl_models_dir",
        default="models_rl",
        help="Directory containing previously trained PPO and DQN models.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set random seeds across common libraries used in the project."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def determine_feature_columns(feature_data: pd.DataFrame) -> list[str]:
    """Select feature columns used by ML and RL models."""

    columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in feature_data.columns]
    if not columns:
        raise ValueError("No engineered feature columns were found in the dataset.")
    return columns


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV with automatic parent directory creation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Save a JSON payload to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_moving_average_test_input(
    full_feature_data: pd.DataFrame,
    test_start_index: int,
    long_window: int,
) -> pd.DataFrame:
    """Provide enough historical context for the moving-average strategy."""

    context_start_index = max(0, test_start_index - (long_window - 1))
    return full_feature_data.iloc[context_start_index:].copy()


def filter_to_test_period(strategy_results: pd.DataFrame, test_start_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only rows belonging to the test period."""

    frame = strategy_results.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame.loc[frame["Date"] >= test_start_date].reset_index(drop=True)


def build_rl_evaluation_input(
    full_feature_data: pd.DataFrame,
    test_start_index: int,
    window_size: int,
) -> pd.DataFrame:
    """Provide enough historical context for RL observations on the test period."""

    context_start_index = max(0, test_start_index - (window_size - 1))
    return full_feature_data.iloc[context_start_index:].copy()


def evaluate_loaded_rl_agent(
    model: Any,
    evaluation_data: pd.DataFrame,
    feature_columns: list[str],
    initial_cash: float,
    transaction_cost: float,
    window_size: int,
) -> pd.DataFrame:
    """Evaluate a loaded RL agent on the supplied market data."""

    from src.rl.trading_env import TradingEnv

    env = TradingEnv(
        data=evaluation_data,
        feature_columns=feature_columns,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        window_size=window_size,
    )
    observation, _ = env.reset()

    terminated = False
    truncated = False
    records: list[dict[str, Any]] = []

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(int(action))
        records.append(
            {
                "Date": pd.Timestamp(info["date"]),
                "Close": float(evaluation_data.iloc[info["step"]]["Close"]),
                "action": int(info["action"]) if info["action"] is not None else 0,
                "position": int(info["position"]),
                "portfolio_value": float(info["portfolio_value"]),
            }
        )

    if not records:
        raise ValueError("RL evaluation produced no records.")

    return pd.DataFrame(records)


def build_metrics_frame_from_rl_results(
    evaluation_results: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """Build a metrics-ready DataFrame from RL evaluation results."""

    if evaluation_results.empty:
        raise ValueError("evaluation_results must not be empty.")

    frame = evaluation_results.copy()
    previous_values = frame["portfolio_value"].shift(1).fillna(initial_cash)
    frame["strategy_return"] = frame["portfolio_value"] / previous_values - 1.0
    return frame.loc[:, ["portfolio_value", "strategy_return", "position"]].copy()


def try_load_and_run_rl_agents(
    ticker: str,
    full_feature_data: pd.DataFrame,
    test_start_index: int,
    test_start_date: pd.Timestamp,
    feature_columns: list[str],
    initial_cash: float,
    transaction_cost: float,
    window_size: int,
    rl_models_dir: Path,
    reports: ReportPaths,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load previously trained PPO and DQN agents if available and evaluate them."""

    rl_results: dict[str, pd.DataFrame] = {}
    messages: list[str] = []

    try:
        from stable_baselines3 import DQN, PPO
    except ImportError:
        messages.append(
            "stable-baselines3 is not installed, so PPO and DQN evaluation was skipped."
        )
        return rl_results, messages

    evaluation_data = build_rl_evaluation_input(
        full_feature_data=full_feature_data,
        test_start_index=test_start_index,
        window_size=window_size,
    )
    model_specs = {
        "ppo_agent": (PPO, rl_models_dir / f"ppo_{ticker.lower()}.zip"),
        "dqn_agent": (DQN, rl_models_dir / f"dqn_{ticker.lower()}.zip"),
    }

    for strategy_name, (model_class, model_path) in model_specs.items():
        if not model_path.exists():
            messages.append(
                f"{strategy_name} was skipped because the model file was not found: {model_path}"
            )
            continue

        try:
            model = model_class.load(str(model_path))
            evaluation_results = evaluate_loaded_rl_agent(
                model=model,
                evaluation_data=evaluation_data,
                feature_columns=feature_columns,
                initial_cash=initial_cash,
                transaction_cost=transaction_cost,
                window_size=window_size,
            )
            evaluation_results = filter_to_test_period(
                strategy_results=evaluation_results,
                test_start_date=test_start_date,
            )
            if evaluation_results.empty:
                messages.append(
                    f"{strategy_name} produced no rows after filtering to the test period."
                )
                continue

            rl_results[strategy_name] = evaluation_results
            save_dataframe(
                evaluation_results,
                reports.agents_dir / f"{strategy_name}_actions.csv",
            )
        except Exception as error:
            messages.append(f"{strategy_name} evaluation was skipped: {error}")

    return rl_results, messages


def calculate_strategy_metrics(
    strategy_results: dict[str, pd.DataFrame],
    initial_cash: float,
) -> pd.DataFrame:
    """Calculate financial metrics for all available strategies."""

    metric_rows: list[dict[str, Any]] = []

    for strategy_name, result_frame in strategy_results.items():
        if {"portfolio_value", "strategy_return", "position"}.issubset(result_frame.columns):
            metrics_frame = result_frame.loc[
                :, ["portfolio_value", "strategy_return", "position"]
            ].copy()
        else:
            metrics_frame = build_metrics_frame_from_rl_results(
                evaluation_results=result_frame,
                initial_cash=initial_cash,
            )

        metrics = calculate_all_metrics(metrics_frame)
        metric_rows.append({"strategy_name": strategy_name, **metrics})

    metrics_table = pd.DataFrame(metric_rows)
    return metrics_table.sort_values(
        by=["sharpe_ratio", "total_return"],
        ascending=False,
    ).reset_index(drop=True)


def build_equity_curve_table(strategy_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge equity curves from all strategies into one comparison table."""

    equity_frames: list[pd.DataFrame] = []

    for strategy_name, result_frame in strategy_results.items():
        if not {"Date", "portfolio_value"}.issubset(result_frame.columns):
            continue

        frame = result_frame.loc[:, ["Date", "portfolio_value"]].copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "portfolio_value"]).sort_values("Date")
        frame = frame.rename(columns={"portfolio_value": strategy_name})
        equity_frames.append(frame)

    if not equity_frames:
        raise ValueError("No strategy equity curves are available for merging.")

    return reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"),
        equity_frames,
    ).sort_values("Date").reset_index(drop=True)


def save_visualizations(
    strategy_results: dict[str, pd.DataFrame],
    metrics_table: pd.DataFrame,
    reports: ReportPaths,
) -> None:
    """Generate and save charts for the experiment results."""

    plot_equity_curves(
        strategy_results=strategy_results,
        title="Strategy Equity Curve Comparison",
        save_path=reports.plots_dir / "equity_curves_comparison.png",
    )

    for strategy_name, result_frame in strategy_results.items():
        plot_price_with_signals(
            strategy_results=result_frame,
            title=f"{strategy_name} Price Signals",
            save_path=reports.plots_dir / f"{strategy_name}_price_signals.png",
        )
        plot_drawdown(
            strategy_results=result_frame,
            title=f"{strategy_name} Drawdown",
            save_path=reports.plots_dir / f"{strategy_name}_drawdown.png",
        )

    for metric_name in ("total_return", "sharpe_ratio", "max_drawdown"):
        plot_metrics_bar(
            metrics_data=metrics_table,
            metric_name=metric_name,
            label_column="strategy_name",
            title=f"{metric_name} Comparison",
            save_path=reports.plots_dir / f"{metric_name}_comparison.png",
        )


def main() -> None:
    """Run the complete comparison pipeline and save experiment outputs."""

    args = parse_args()
    set_global_seed(args.random_state)

    ticker = args.ticker.strip().upper()
    if not ticker:
        raise ValueError("--ticker must not be empty.")

    run_name = f"{ticker.lower()}_{args.start_date}_{args.end_date}"
    reports = ReportPaths.create(Path(args.reports_dir), run_name)

    raw_data_path = reports.data_dir / f"{ticker.lower()}_raw.csv"
    feature_data_path = reports.data_dir / f"{ticker.lower()}_features.csv"
    metadata_path = reports.run_dir / "run_metadata.json"

    raw_data = download_ohlcv(
        ticker=ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        save_path=str(raw_data_path),
    )
    feature_data = build_feature_set(raw_data)
    if feature_data.empty:
        raise ValueError("Feature generation returned an empty dataset.")

    save_dataframe(feature_data, feature_data_path)
    feature_columns = determine_feature_columns(feature_data)
    train_df, val_df, test_df = split_time_series(
        feature_data,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    test_start_index = int(test_df.index[0])
    test_start_date = pd.Timestamp(test_df.iloc[0]["Date"])

    buy_hold_results = BuyAndHoldStrategy(
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
    ).run(test_df.loc[:, ["Date", "Close"]].copy())

    moving_average_input = build_moving_average_test_input(
        full_feature_data=feature_data,
        test_start_index=test_start_index,
        long_window=args.long_window,
    )
    moving_average_results = MovingAverageCrossoverStrategy(
        short_window=args.short_window,
        long_window=args.long_window,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
    ).run(moving_average_input.loc[:, ["Date", "Close"]].copy())
    moving_average_results = filter_to_test_period(
        strategy_results=moving_average_results,
        test_start_date=test_start_date,
    )

    ml_comparison_table, ml_strategy_results, _ = train_and_evaluate_models(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_columns=feature_columns,
        target_column="target_direction",
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        random_state=args.random_state,
    )
    save_dataframe(ml_comparison_table, reports.metrics_dir / "ml_classification_metrics.csv")

    for model_name, prediction_frame in ml_strategy_results.items():
        save_dataframe(
            prediction_frame,
            reports.predictions_dir / f"{model_name}_predictions.csv",
        )

    strategy_results: dict[str, pd.DataFrame] = {
        "buy_hold": buy_hold_results,
        "moving_average": moving_average_results,
        **ml_strategy_results,
    }

    save_dataframe(buy_hold_results, reports.strategies_dir / "buy_hold_results.csv")
    save_dataframe(
        moving_average_results,
        reports.strategies_dir / "moving_average_results.csv",
    )

    rl_results, rl_messages = try_load_and_run_rl_agents(
        ticker=ticker,
        full_feature_data=feature_data,
        test_start_index=test_start_index,
        test_start_date=test_start_date,
        feature_columns=feature_columns,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        window_size=args.rl_window_size,
        rl_models_dir=Path(args.rl_models_dir),
        reports=reports,
    )
    strategy_results.update(rl_results)

    metrics_table = calculate_strategy_metrics(
        strategy_results=strategy_results,
        initial_cash=args.initial_cash,
    )
    save_dataframe(metrics_table, reports.metrics_dir / "all_strategies_metrics.csv")

    equity_curve_table = build_equity_curve_table(strategy_results)
    save_dataframe(equity_curve_table, reports.equity_dir / "all_equity_curves.csv")

    save_visualizations(
        strategy_results=strategy_results,
        metrics_table=metrics_table,
        reports=reports,
    )

    metadata = {
        "ticker": ticker,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_cash": args.initial_cash,
        "transaction_cost": args.transaction_cost,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "short_window": args.short_window,
        "long_window": args.long_window,
        "rl_window_size": args.rl_window_size,
        "feature_columns": feature_columns,
        "available_strategies": list(strategy_results.keys()),
        "rl_messages": rl_messages,
    }
    save_json(metadata, metadata_path)

    print("Experiment completed successfully.")
    print(f"Run directory: {reports.run_dir}")
    print(f"Saved metrics table: {reports.metrics_dir / 'all_strategies_metrics.csv'}")
    print(f"Saved equity curves: {reports.equity_dir / 'all_equity_curves.csv'}")
    print(f"Saved plots in: {reports.plots_dir}")
    print(f"Saved predictions in: {reports.predictions_dir}")
    print(f"Saved RL actions in: {reports.agents_dir}")
    if rl_messages:
        print("RL status:")
        for message in rl_messages:
            print(f"  - {message}")


if __name__ == "__main__":
    main()
