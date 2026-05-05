"""Evaluate a trained RL agent on the trading environment."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.metrics.financial_metrics import calculate_financial_metrics
from src.rl.trading_env import TradingEnv


class PredictableModel(Protocol):
    """Minimal protocol for Stable-Baselines3-like models."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]:
        """Return an action for the provided observation."""


def evaluate_agent(
    model: PredictableModel,
    prices: np.ndarray,
    features: np.ndarray,
) -> dict[str, float]:
    """Run one deterministic evaluation episode and compute key metrics."""

    env = TradingEnv(prices=prices, features=features)
    observation, _ = env.reset()

    terminated = False
    truncated = False
    rewards: list[float] = []

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(int(action))
        rewards.append(float(reward))

    reward_series = np.asarray(rewards, dtype=float)
    metrics = calculate_financial_metrics(returns=reward_series)
    metrics["final_equity"] = float(env.cash)
    return metrics
