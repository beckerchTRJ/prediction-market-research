from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalshi_fund.panel import clip_prob, inv_logit, logit_prob


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    ridge_alpha: float = 1.0
    min_train_rows: int = 250
    target_col: str = "target_correction"
    probability_target_col: str = "objective_probability"


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    ridge_alpha: float,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=ridge_alpha))])


def _time_splits(frame: pd.DataFrame, config: WalkForwardConfig) -> list[tuple[pd.Index, pd.Index, pd.Timestamp, pd.Timestamp]]:
    timeline = pd.to_datetime(frame["timestamp"]).dt.normalize()
    min_date = timeline.min()
    max_date = timeline.max()
    train_end = min_date + timedelta(days=config.min_train_days)
    splits: list[tuple[pd.Index, pd.Index, pd.Timestamp, pd.Timestamp]] = []

    while train_end < max_date:
        test_end = train_end + timedelta(days=config.test_days)
        train_idx = frame.index[timeline < train_end]
        test_idx = frame.index[(timeline >= train_end) & (timeline < test_end)]
        if len(train_idx) >= config.min_train_rows and len(test_idx) > 0:
            splits.append((train_idx, test_idx, train_end, test_end))
        train_end = train_end + timedelta(days=config.step_days)

    return splits


def walk_forward_predictions(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["timestamp", "race_id", "contract", "cycle", "p_mkt", config.target_col, config.probability_target_col]
    filtered = frame.dropna(subset=required).copy()
    if filtered.empty:
        msg = "no rows remain after filtering for target and probability columns"
        raise ValueError(msg)

    splits = _time_splits(filtered, config)
    if not splits:
        msg = "not enough history to construct walk-forward splits"
        raise ValueError(msg)

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, float | int | str]] = []

    for fold_number, (train_idx, test_idx, train_end, test_end) in enumerate(splits, start=1):
        model = build_model_pipeline(numeric_features, categorical_features, config.ridge_alpha)
        train_frame = filtered.loc[train_idx]
        test_frame = filtered.loc[test_idx].copy()

        model.fit(train_frame[numeric_features + categorical_features], train_frame[config.target_col])
        test_frame["delta_hat"] = model.predict(test_frame[numeric_features + categorical_features])
        test_frame["p_fair_model"] = clip_prob(inv_logit(logit_prob(test_frame["p_mkt"]) + test_frame["delta_hat"]))
        test_frame["fold"] = fold_number
        test_frame["train_end"] = train_end
        test_frame["test_end"] = test_end

        objective = test_frame[config.probability_target_col]
        market_rmse = float(np.sqrt(np.mean(np.square(objective - test_frame["p_mkt"]))))
        model_rmse = float(np.sqrt(np.mean(np.square(objective - test_frame["p_fair_model"]))))
        correction_rmse = float(np.sqrt(np.mean(np.square(test_frame[config.target_col] - test_frame["delta_hat"]))))

        metrics.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "train_end": train_end.isoformat(),
                "test_end": test_end.isoformat(),
                "market_prob_rmse": market_rmse,
                "model_prob_rmse": model_rmse,
                "target_correction_rmse": correction_rmse,
            }
        )
        predictions.append(test_frame)

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def estimate_trade_parameters(
    predictions: pd.DataFrame,
    probability_target_col: str = "objective_probability",
) -> pd.DataFrame:
    frame = predictions.dropna(subset=["p_mkt", "p_fair_model", probability_target_col]).copy()
    if frame.empty:
        msg = "predictions are missing required probability columns"
        raise ValueError(msg)

    model_delta = frame["p_fair_model"] - frame["p_mkt"]
    truth_delta = frame[probability_target_col] - frame["p_mkt"]
    denominator = float(np.square(model_delta).sum())
    alpha = 0.0 if denominator == 0.0 else float(np.clip((model_delta * truth_delta).sum() / denominator, 0.0, 1.0))
    frame["p_blend"] = clip_prob(frame["p_mkt"] + alpha * model_delta)
    sigma = float(np.sqrt(np.mean(np.square(frame[probability_target_col] - frame["p_blend"]))))

    summary = {
        "alpha": alpha,
        "sigma": sigma,
        "oof_rows": int(len(frame)),
        "market_prob_rmse": float(np.sqrt(np.mean(np.square(frame[probability_target_col] - frame["p_mkt"])))),
        "model_prob_rmse": float(np.sqrt(np.mean(np.square(frame[probability_target_col] - frame["p_fair_model"])))),
        "blend_prob_rmse": float(np.sqrt(np.mean(np.square(frame[probability_target_col] - frame["p_blend"])))),
    }
    return pd.DataFrame([summary])


def fit_full_sample_predictions(
    train_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    filtered_train = train_frame.dropna(subset=["p_mkt", config.target_col]).copy()
    if filtered_train.empty:
        msg = "training frame is missing rows with targets"
        raise ValueError(msg)

    model = build_model_pipeline(numeric_features, categorical_features, config.ridge_alpha)
    feature_columns = numeric_features + categorical_features
    model.fit(filtered_train[feature_columns], filtered_train[config.target_col])

    scored = score_frame.copy()
    scored["delta_hat"] = model.predict(scored[feature_columns])
    scored["p_fair_model"] = clip_prob(inv_logit(logit_prob(scored["p_mkt"]) + scored["delta_hat"]))
    return scored
