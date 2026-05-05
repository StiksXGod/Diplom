"""Machine-learning baselines for forecasting price direction."""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_TEST_COLUMNS: Final[tuple[str, str]] = ("Date", "Close")


def build_model_registry(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Create baseline classifiers for direction prediction."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def _validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    split_name: str,
    require_test_columns: bool = False,
) -> pd.DataFrame:
    """Validate one dataset split and return a cleaned copy."""

    if df.empty:
        raise ValueError(f"{split_name} must not be empty.")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    required_columns = list(feature_columns) + [target_column]
    if require_test_columns:
        required_columns.extend(REQUIRED_TEST_COLUMNS)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"{split_name} is missing required columns: {missing}.")

    frame = df.copy()
    numeric_columns = list(feature_columns) + [target_column]
    if require_test_columns:
        numeric_columns.append("Close")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")

    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required_columns).reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"{split_name} contains no valid rows after cleaning.")

    target_values = set(frame[target_column].astype(int).unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{target_column} in {split_name} must contain only binary labels 0 and 1.")

    frame[target_column] = frame[target_column].astype(int)
    if require_test_columns and (frame["Close"] <= 0).any():
        raise ValueError(f"Close prices in {split_name} must be strictly positive.")

    return frame


def _extract_xy(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target vector from a split."""

    x = df.loc[:, list(feature_columns)].copy()
    y = df[target_column].astype(int).copy()
    return x, y


def _predict_probabilities(model: ClassifierMixin, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for the provided samples."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        return np.asarray(probabilities[:, 1], dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = np.asarray(model.predict(x), dtype=float)
    return predictions


def evaluate_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics for one model."""

    roc_auc = 0.0
    if y_true.nunique() > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            roc_auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
    }


def convert_predictions_to_strategy(
    df: pd.DataFrame,
    prediction_column: str = "prediction",
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """Convert binary model predictions into a simple long-flat strategy.

    The prediction at time ``t`` is interpreted as a trading signal for the
    next period return. If ``target_return`` exists in the input DataFrame, it is
    used directly; otherwise the forward return is computed from ``Close``.
    """

    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive.")
    if not 0.0 <= transaction_cost < 1.0:
        raise ValueError("transaction_cost must be in the [0, 1) interval.")
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    required_columns = list(REQUIRED_TEST_COLUMNS) + [prediction_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"Input DataFrame is missing required columns: {missing}.")

    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame[prediction_column] = pd.to_numeric(frame[prediction_column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required_columns).sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise ValueError("Input DataFrame contains no valid rows after cleaning.")
    if (frame["Close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")

    frame["position"] = (frame[prediction_column].astype(int) == 1).astype(int)

    if "target_return" in frame.columns:
        frame["daily_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
    else:
        frame["daily_return"] = frame["Close"].shift(-1) / frame["Close"] - 1.0

    frame["daily_return"] = frame["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover = frame["position"].diff().abs().fillna(frame["position"])
    frame["strategy_return"] = frame["position"] * frame["daily_return"] - (
        turnover * transaction_cost
    )
    frame["portfolio_value"] = initial_cash * (1.0 + frame["strategy_return"]).cumprod()

    return frame


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = "target_direction",
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, ClassifierMixin]]:
    """Train baseline models, evaluate them on the test split, and save predictions.

    Args:
        train_df: Training split with features and target.
        val_df: Validation split. It is merged with train data for final fitting
            because the baseline models use fixed hyperparameters.
        test_df: Test split used only for final evaluation.
        feature_columns: List of feature column names.
        target_column: Binary target column, defaults to ``target_direction``.
        initial_cash: Initial portfolio value for strategy simulation.
        transaction_cost: Cost paid when the predicted position changes.
        random_state: Random seed for reproducible model training.

    Returns:
        Tuple with:
        1. comparison table of classification metrics,
        2. dictionary of test DataFrames with saved predictions and strategy path,
        3. dictionary of trained fitted models.
    """

    prepared_train = _validate_feature_columns(
        df=train_df,
        feature_columns=feature_columns,
        target_column=target_column,
        split_name="train_df",
    )
    prepared_val = _validate_feature_columns(
        df=val_df,
        feature_columns=feature_columns,
        target_column=target_column,
        split_name="val_df",
    )
    prepared_test = _validate_feature_columns(
        df=test_df,
        feature_columns=feature_columns,
        target_column=target_column,
        split_name="test_df",
        require_test_columns=True,
    )

    fit_df = pd.concat([prepared_train, prepared_val], ignore_index=True)
    x_fit, y_fit = _extract_xy(
        df=fit_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    x_test, y_test = _extract_xy(
        df=prepared_test,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    model_registry = build_model_registry(random_state=random_state)
    comparison_rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    trained_models: dict[str, ClassifierMixin] = {}

    for model_name, model in model_registry.items():
        fitted_model = clone(model)
        fitted_model.fit(x_fit, y_fit)

        y_pred = np.asarray(fitted_model.predict(x_test), dtype=int)
        y_proba = _predict_probabilities(fitted_model, x_test)
        metrics = evaluate_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        comparison_rows.append({"model_name": model_name, **metrics})

        prediction_frame = prepared_test.copy()
        prediction_frame["prediction"] = y_pred
        prediction_frame["prediction_proba"] = y_proba
        predictions_by_model[model_name] = convert_predictions_to_strategy(
            df=prediction_frame,
            prediction_column="prediction",
            initial_cash=initial_cash,
            transaction_cost=transaction_cost,
        )
        trained_models[model_name] = fitted_model

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table = comparison_table.sort_values(
        by=["f1", "roc_auc", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison_table, predictions_by_model, trained_models
