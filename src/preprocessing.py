"""
Primary preprocessing for the PIRvision study.

Primary experiment:
    PIR sensors only.

Temperature_F is intentionally excluded from the model feature matrix because
its zero values perfectly identify the motion/activity class in the supplied
dataset. It is retained in the raw dataframe for audit and optional future
sensitivity analysis, but it is NEVER passed to the primary model.

All scaling is performed inside sklearn Pipelines so it is learned only from
training folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CANONICAL_CSV,
    TARGET_COL,
    ORIG_TO_ENC,
    TEMPORAL_WINDOW_SECONDS,
    get_logger,
)
from src.utils import detect_pir_columns, build_temporal_groups

logger = get_logger("preprocessing")

NEEDS_SCALING = {"LogisticRegression", "SVM", "KNN", "MLP"}


@dataclass
class Dataset:
    df: pd.DataFrame
    X: pd.DataFrame
    y_original: np.ndarray
    y_encoded: np.ndarray
    groups: np.ndarray
    pir_cols: list[str]
    feature_cols: list[str]


def load_dataset(window_seconds: int | None = None) -> Dataset:
    if window_seconds is None:
        window_seconds = TEMPORAL_WINDOW_SECONDS

    df = pd.read_csv(CANONICAL_CSV)

    required = {"Date", "Time", TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing)}"
        )

    df = df.sort_values(["Date", "Time"]).reset_index(drop=True)

    pir_cols = detect_pir_columns(df)
    if not pir_cols:
        raise ValueError("No PIR_* sensor columns were detected.")

    # IMPORTANT: primary model input is PIR ONLY.
    feature_cols = list(pir_cols)
    X = df[feature_cols].copy()

    y_original = df[TARGET_COL].to_numpy()

    unexpected = set(np.unique(y_original)) - set(ORIG_TO_ENC)
    if unexpected:
        raise ValueError(
            f"Unexpected target values found: {sorted(unexpected)}"
        )

    y_encoded = (
        df[TARGET_COL]
        .map(ORIG_TO_ENC)
        .to_numpy(dtype=int)
    )

    groups = build_temporal_groups(
        df,
        window_seconds=window_seconds,
    )

    logger.info(
        "Primary feature policy: PIR sensors only. "
        f"n_pir={len(pir_cols)}, "
        f"X_shape={X.shape}, "
        f"n_groups={len(np.unique(groups))}"
    )
    logger.info(
        "Temperature_F remains available in the raw dataframe for audit "
        "but is excluded from the primary model feature matrix."
    )

    return Dataset(
        df=df,
        X=X,
        y_original=y_original,
        y_encoded=y_encoded,
        groups=groups,
        pir_cols=pir_cols,
        feature_cols=feature_cols,
    )


def make_pipeline(model_name: str, estimator) -> Pipeline:
    if model_name in NEEDS_SCALING:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    return Pipeline([
        ("model", estimator),
    ])


if __name__ == "__main__":
    ds = load_dataset()
    print(ds.X.shape, ds.feature_cols[:5], ds.y_original[:5])
