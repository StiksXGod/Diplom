"""Train and evaluate a DQN trading agent from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data.features import DEFAULT_FEATURE_COLUMNS
from src.metrics.financial_metrics import calculate_all_metrics
from src.rl.trading_env import TradingEnv


DEFAULT_TRAIN_SIZE: Final[float] = 0.8
DEFAULT_WINDOW_SIZE: Final[int] = 30
DEFAULT_PERIODS_PER_YEAR: Final[int] = 252
OUTPUT_DIR = Path("models_rl")


def load_feature_data(data_path: str | Path) -> pd.DataFrame:
    """Load a prepared feature CSV file and validate core columns."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("Loaded CSV file is empty.")

    required_columns = {"Date", "Close"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input CSV is missing required columns: {missing}.")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid rows remain after cleaning Date and Close columns.")
    if (data["Close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")

    return data


def infer_feature_columns(data: pd.DataFrame) -> list[str]:
    """Infer feature columns from the prepared dataset."""

    preferred_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in data.columns]
    if preferred_columns:
        return preferred_columns

    excluded_columns = {
        "Date",
        "target_direction",
        "target_return",
        "portfolio_value",
        "strategy_return",
        "position",
        "action",
    }
    feature_columns: list[str] = []

    for column in data.columns:
        if column in excluded_columns:
            continue
        numeric_series = pd.to_numeric(data[column], errors="coerce")
        if numeric_series.notna().all():
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("Could not infer numeric feature columns from the input dataset.")

    return feature_columns


def split_train_test_by_time(
    data: pd.DataFrame,
    train_size: float = DEFAULT_TRAIN_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into chronological train and test subsets."""

    if data.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be in the (0, 1) interval.")

    split_index = int(len(data) * train_size)
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("train_size creates an empty train or test split.")

    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()
    return train_df, test_df


def make_dqn_env(
    data: pd.DataFrame,
    feature_columns: list[str],
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> DummyVecEnv:
    """Wrap TradingEnv for Stable-Baselines3 DQN training."""

    return DummyVecEnv(
        [
            lambda: TradingEnv(
                data=data,
                feature_columns=feature_columns,
                initial_cash=initial_cash,
                transaction_cost=transaction_cost,
                window_size=window_size,
            )
        ]
    )


def train_dqn(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    total_timesteps: int = 10_000,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> DQN:
    """Train a DQN agent on the training subset."""

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")

    env = make_dqn_env(
        data=train_df,
        feature_columns=feature_columns,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        window_size=window_size,
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=5_000,
        learning_starts=500,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_dqn_agent(
    model: DQN,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> pd.DataFrame:
    """Run the trained DQN agent on the test environment and collect results."""

    env = TradingEnv(
        data=test_df,
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
                "Close": float(test_df.iloc[info["step"]]["Close"]),
                "action": int(info["action"]) if info["action"] is not None else 0,
                "position": int(info["position"]),
                "portfolio_value": float(info["portfolio_value"]),
            }
        )

    if not records:
        raise ValueError("Evaluation produced no records. The test split may be too short.")

    return pd.DataFrame(records)


def build_metrics_frame(
    evaluation_results: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """Prepare evaluation output for financial metrics calculation."""

    if evaluation_results.empty:
        raise ValueError("evaluation_results must not be empty.")

    frame = evaluation_results.copy()
    previous_values = frame["portfolio_value"].shift(1).fillna(initial_cash)
    frame["strategy_return"] = frame["portfolio_value"] / previous_values - 1.0
    return frame.loc[:, ["portfolio_value", "strategy_return", "position"]].copy()


def save_metrics(
    metrics: dict[str, float | int],
    csv_path: Path,
    json_path: Path,
) -> None:
    """Persist metrics in CSV and JSON formats."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def save_evaluation_results(results: pd.DataFrame, output_path: Path) -> None:
    """Save test evaluation records, including actions and equity curve, to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DQN training."""

    parser = argparse.ArgumentParser(
        description="Train a DQN trading agent on prepared feature data."
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the prepared CSV file with engineered features.",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol used for naming saved artifacts.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Number of DQN training timesteps.",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=10_000.0,
        help="Initial portfolio cash for the environment.",
    )
    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.001,
        help="Transaction cost applied on buy and sell operations.",
    )
    return parser.parse_args()


def main() -> None:
    """Train DQN on a prepared dataset and save evaluation artifacts."""

    args = parse_args()
    ticker = args.ticker.strip().upper()
    if not ticker:
        raise ValueError("--ticker must not be empty.")

    data = load_feature_data(args.data_path)
    feature_columns = infer_feature_columns(data)
    train_df, test_df = split_train_test_by_time(data)

    if len(train_df) <= DEFAULT_WINDOW_SIZE:
        raise ValueError("Train split is too short for the configured window_size.")
    if len(test_df) <= DEFAULT_WINDOW_SIZE:
        raise ValueError("Test split is too short for the configured window_size.")

    model = train_dqn(
        train_df=train_df,
        feature_columns=feature_columns,
        total_timesteps=args.timesteps,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        window_size=DEFAULT_WINDOW_SIZE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"dqn_{ticker.lower()}.zip"
    model.save(str(model_path))

    evaluation_results = evaluate_dqn_agent(
        model=model,
        test_df=test_df,
        feature_columns=feature_columns,
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        window_size=DEFAULT_WINDOW_SIZE,
    )
    results_path = OUTPUT_DIR / f"dqn_{ticker.lower()}_test_results.csv"
    save_evaluation_results(evaluation_results, results_path)

    metrics_frame = build_metrics_frame(
        evaluation_results=evaluation_results,
        initial_cash=args.initial_cash,
    )
    metrics = calculate_all_metrics(
        df=metrics_frame,
        periods_per_year=DEFAULT_PERIODS_PER_YEAR,
    )

    metrics_csv_path = OUTPUT_DIR / f"dqn_{ticker.lower()}_metrics.csv"
    metrics_json_path = OUTPUT_DIR / f"dqn_{ticker.lower()}_metrics.json"
    save_metrics(metrics=metrics, csv_path=metrics_csv_path, json_path=metrics_json_path)

    print(f"Training completed for {ticker}.")
    print(f"Feature columns: {', '.join(feature_columns)}")
    print(f"Model saved to: {model_path}")
    print(f"Test results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_csv_path} and {metrics_json_path}")
    print("Metrics summary:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
