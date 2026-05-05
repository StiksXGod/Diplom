"""Gymnasium trading environment for a single-asset long-only strategy."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Final

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


EPSILON: Final[float] = 1e-12


class TradingAction(IntEnum):
    """Discrete actions supported by the trading environment."""

    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnv(gym.Env[np.ndarray, int]):
    """Custom Gymnasium environment for trading one asset.

    The environment is long-only and operates on close prices. Observations are
    formed as a flattened window of recent feature values followed by the
    current position flag and the fraction of portfolio value held in cash.

    For backward compatibility with the rest of the project, the environment can
    also be initialized from ``prices`` and ``features`` arrays, although the
    primary interface uses a pandas DataFrame and feature column names.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        feature_columns: Sequence[str] | None = None,
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 30,
        *,
        prices: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.window_size = int(window_size)
        self._validate_parameters()

        self.data, self.feature_columns = self._prepare_data(
            data=data,
            feature_columns=feature_columns,
            prices=prices,
            features=features,
        )
        if len(self.data) <= self.window_size:
            raise ValueError("Input data must contain more rows than window_size.")

        self.close_prices = self.data["Close"].to_numpy(dtype=np.float32)
        self.feature_matrix = self.data.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        self.dates = self.data["Date"].tolist()

        observation_size = self.window_size * len(self.feature_columns) + 2
        self.action_space = spaces.Discrete(len(TradingAction))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.shares_held = 0.0
        self.position = 0
        self.portfolio_value = self.initial_cash

        self.actions: list[int] = []
        self.portfolio_values: list[float] = []
        self.positions: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment state and return the first observation."""

        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.shares_held = 0.0
        self.position = 0
        self.portfolio_value = self._get_portfolio_value(
            price=float(self.close_prices[self.current_step])
        )

        self.actions = []
        self.portfolio_values = [self.portfolio_value]
        self.positions = [self.position]

        return self._get_observation(), self._build_info(action=None, reward=0.0, commission=0.0)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action and advance the environment by one time step."""

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Allowed actions: {list(TradingAction)}.")
        if self.current_step >= len(self.data) - 1:
            return (
                self._get_observation(),
                0.0,
                True,
                False,
                self._build_info(action=action, reward=0.0, commission=0.0),
            )

        current_price = float(self.close_prices[self.current_step])
        previous_portfolio_value = self._get_portfolio_value(price=current_price)
        commission = self._execute_action(action=TradingAction(action), price=current_price)

        self.current_step += 1
        next_price = float(self.close_prices[self.current_step])
        self.portfolio_value = self._get_portfolio_value(price=next_price)
        reward = float(self.portfolio_value - previous_portfolio_value)

        self.actions.append(int(action))
        self.portfolio_values.append(self.portfolio_value)
        self.positions.append(self.position)

        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = self._build_info(action=action, reward=reward, commission=commission)

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable summary of the current environment state."""

        current_price = float(self.close_prices[self.current_step])
        current_date = self.dates[self.current_step]
        print(
            f"date={current_date} step={self.current_step} price={current_price:.2f} "
            f"position={self.position} shares={self.shares_held:.6f} "
            f"cash={self.cash:.2f} portfolio_value={self.portfolio_value:.2f}"
        )

    @property
    def history(self) -> dict[str, list[int] | list[float]]:
        """Expose tracked history for actions, positions, and portfolio value."""

        return {
            "actions": self.actions,
            "portfolio_values": self.portfolio_values,
            "positions": self.positions,
        }

    def _validate_parameters(self) -> None:
        """Validate environment configuration."""

        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if not 0.0 <= self.transaction_cost < 1.0:
            raise ValueError("transaction_cost must be in the [0, 1) interval.")
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

    def _prepare_data(
        self,
        data: pd.DataFrame | None,
        feature_columns: Sequence[str] | None,
        prices: np.ndarray | None,
        features: np.ndarray | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Prepare and validate the environment input data."""

        if data is not None:
            return self._prepare_from_dataframe(data=data, feature_columns=feature_columns)
        if prices is not None and features is not None:
            return self._prepare_from_arrays(prices=prices, features=features)

        raise ValueError(
            "Provide either a DataFrame with feature_columns or both prices and features arrays."
        )

    def _prepare_from_dataframe(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Validate a DataFrame-based dataset."""

        if data.empty:
            raise ValueError("Input DataFrame must not be empty.")
        if feature_columns is None or len(feature_columns) == 0:
            raise ValueError("feature_columns must not be empty.")

        selected_feature_columns = list(feature_columns)
        required_columns = ["Close", *selected_feature_columns]
        if "Date" in data.columns:
            required_columns = ["Date", *required_columns]

        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

        frame = data.loc[:, required_columns].copy()
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame = frame.sort_values("Date")
        else:
            frame.insert(0, "Date", pd.RangeIndex(start=0, stop=len(frame)))

        numeric_columns = ["Close", *selected_feature_columns]
        frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["Date", *numeric_columns]).reset_index(drop=True)

        if frame.empty:
            raise ValueError("Input DataFrame contains no valid rows after cleaning.")
        if (frame["Close"] <= 0).any():
            raise ValueError("Close prices must be strictly positive.")

        return frame, selected_feature_columns

    def _prepare_from_arrays(
        self,
        prices: np.ndarray,
        features: np.ndarray,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build a DataFrame from legacy price and feature arrays."""

        prices_array = np.asarray(prices, dtype=float)
        features_array = np.asarray(features, dtype=float)

        if prices_array.ndim != 1:
            raise ValueError("prices must be a one-dimensional array.")
        if features_array.ndim != 2:
            raise ValueError("features must be a two-dimensional array.")
        if len(prices_array) != len(features_array):
            raise ValueError("prices and features must have the same number of rows.")
        if len(prices_array) == 0:
            raise ValueError("prices and features must not be empty.")

        feature_columns = [f"feature_{index}" for index in range(features_array.shape[1])]
        frame = pd.DataFrame(features_array, columns=feature_columns)
        frame.insert(0, "Date", pd.RangeIndex(start=0, stop=len(frame)))
        frame["Close"] = prices_array

        return self._prepare_from_dataframe(data=frame, feature_columns=feature_columns)

    def _execute_action(self, action: TradingAction, price: float) -> float:
        """Execute the requested trading action and return the paid commission."""

        commission = 0.0

        if action == TradingAction.BUY and self.position == 0 and self.cash > EPSILON:
            shares_to_buy = self.cash / (price * (1.0 + self.transaction_cost))
            if shares_to_buy > 0.0:
                gross_cost = shares_to_buy * price
                commission = gross_cost * self.transaction_cost
                total_cost = gross_cost + commission
                self.cash = max(self.cash - total_cost, 0.0)
                self.shares_held = shares_to_buy
                self.position = 1

        elif action == TradingAction.SELL and self.position == 1 and self.shares_held > EPSILON:
            gross_proceeds = self.shares_held * price
            commission = gross_proceeds * self.transaction_cost
            self.cash += gross_proceeds - commission
            self.shares_held = 0.0
            self.position = 0

        return float(commission)

    def _get_observation(self) -> np.ndarray:
        """Build the flattened observation vector for the current step."""

        start_index = self.current_step - self.window_size + 1
        end_index = self.current_step + 1
        feature_window = self.feature_matrix[start_index:end_index]
        flattened_window = feature_window.reshape(-1)

        current_price = float(self.close_prices[self.current_step])
        current_portfolio_value = self._get_portfolio_value(price=current_price)
        cash_fraction = self.cash / current_portfolio_value if current_portfolio_value > EPSILON else 0.0

        observation = np.concatenate(
            [
                flattened_window,
                np.array([float(self.position), float(cash_fraction)], dtype=np.float32),
            ]
        )
        return observation.astype(np.float32)

    def _get_portfolio_value(self, price: float) -> float:
        """Calculate current total portfolio value."""

        return float(self.cash + self.shares_held * price)

    def _build_info(
        self,
        action: int | None,
        reward: float,
        commission: float,
    ) -> dict[str, Any]:
        """Build a Gym-compatible info dictionary."""

        return {
            "date": self.dates[self.current_step],
            "step": self.current_step,
            "action": action,
            "position": self.position,
            "cash": float(self.cash),
            "shares_held": float(self.shares_held),
            "portfolio_value": float(self.portfolio_value),
            "reward": float(reward),
            "commission": float(commission),
        }
