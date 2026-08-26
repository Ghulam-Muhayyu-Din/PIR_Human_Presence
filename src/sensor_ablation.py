"""
PIR sensor ablation for the primary PIR only experiment.

Important methodological design:

For every CV fold:
    1. Fit the selected model on the training fold using all available PIR sensors.
    2. Derive a training fold feature ranking:
       tree feature_importances_ when available, otherwise permutation importance
       evaluated only on the training fold.
    3. Select the top K PIR sensors from that training fold.
    4. Refit the selected model using only those K sensors.
    5. Evaluate on the untouched validation fold.

This prevents the feature ranking itself from being derived from validation data.

Temperature is intentionally absent from every ablation configuration.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    SEED,
    N_SPLITS,
    RESULTS_DIR,
    PIR_SUBSET_SIZES,
    get_logger,
)
from src.model_training import (
    build_model_registry,
    build_pipeline_for,
)

logger = get_logger("sensor_ablation")


def _rank_features_from_training_model(
    pipe,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: list[str],
):
    model = pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(
            model.feature_importances_
        )

    elif hasattr(model, "coef_"):
        coef = np.asarray(
            model.coef_
        )

        if coef.ndim == 1:
            scores = np.abs(coef)

        else:
            scores = np.mean(
                np.abs(coef),
                axis=0,
            )

    else:
        result = permutation_importance(
            pipe,
            X_train,
            y_train,
            n_repeats=5,
            random_state=SEED,
            scoring="f1_macro",
            n_jobs=-1,
        )
        scores = result.importances_mean

    if len(scores) != len(feature_names):
        raise RuntimeError(
            "Feature importance length does not match "
            "the number of PIR sensors."
        )

    ranking = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": scores,
        })
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return ranking


def run_sensor_ablation(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    groups: np.ndarray,
    pir_cols: list,
    importance_df: pd.DataFrame,
    best_model_name: str,
    best_params: dict,
    n_jobs: int = -1,
):

    # Primary experiment must be PIR only.
    non_pir_X = [
        c for c in X.columns
        if c not in pir_cols
    ]

    if non_pir_X:
        raise ValueError(
            "Primary ablation input contains non PIR columns: "
            f"{non_pir_X}"
        )

    if set(X.columns) != set(pir_cols):
        raise ValueError(
            "X columns and PIR columns are inconsistent."
        )

    registry = build_model_registry(
        n_jobs=n_jobs
    )

    if best_model_name not in registry:
        raise KeyError(
            f"Best model '{best_model_name}' is not in model registry."
        )

    spec = registry[
        best_model_name
    ]

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_splits = list(
        sgkf.split(
            X,
            y_encoded,
            groups,
        )
    )

    rows = []

    for k in PIR_SUBSET_SIZES:

        k = min(
            int(k),
            len(pir_cols),
        )

        if k <= 0:
            continue

        fold_macro_f1 = []
        fold_acc = []
        fold_mcc = []
        fit_times = []
        pred_times = []
        selected_lists = []

        logger.info(
            f"Starting nested sensor ablation: K={k}"
        )

        for fold_number, (
            tr_idx,
            va_idx,
        ) in enumerate(
            fold_splits,
            start=1,
        ):

            X_tr_all = X.iloc[
                tr_idx
            ]

            X_va_all = X.iloc[
                va_idx
            ]

            y_tr = y_encoded[
                tr_idx
            ]

            y_va = y_encoded[
                va_idx
            ]

            # Step 1: train on all PIR sensors using training fold only.
            ranking_estimator = (
                spec[
                    "estimator_fn"
                ](**best_params)
            )

            ranking_pipe = (
                build_pipeline_for(
                    best_model_name,
                    ranking_estimator,
                )
            )

            ranking_pipe.fit(
                X_tr_all,
                y_tr,
            )

            ranking = (
                _rank_features_from_training_model(
                    ranking_pipe,
                    X_tr_all,
                    y_tr,
                    pir_cols,
                )
            )

            top_k = ranking[
                "feature"
            ].head(k).tolist()

            selected_lists.append(
                ",".join(top_k)
            )

            # Step 2: refit on only top K sensors.
            X_tr = X_tr_all[
                top_k
            ]

            X_va = X_va_all[
                top_k
            ]

            estimator = (
                spec[
                    "estimator_fn"
                ](**best_params)
            )

            pipe = (
                build_pipeline_for(
                    best_model_name,
                    estimator,
                )
            )

            fit_start = (
                time.perf_counter()
            )

            pipe.fit(
                X_tr,
                y_tr,
            )

            fit_times.append(
                time.perf_counter()
                - fit_start
            )

            pred_start = (
                time.perf_counter()
            )

            y_pred = np.asarray(
                pipe.predict(
                    X_va
                )
            ).reshape(-1)

            pred_times.append(
                time.perf_counter()
                - pred_start
            )

            _, _, f1m, _ = (
                precision_recall_fscore_support(
                    y_va,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            )

            fold_macro_f1.append(
                f1m
            )

            fold_acc.append(
                accuracy_score(
                    y_va,
                    y_pred,
                )
            )

            fold_mcc.append(
                matthews_corrcoef(
                    y_va,
                    y_pred,
                )
            )

            logger.info(
                f"K={k}, fold={fold_number}: "
                f"Macro F1={f1m:.4f}"
            )

        rows.append({
            "n_pir_sensors": k,
            "feature_ranking_protocol": (
                "Fold specific ranking from training data only"
            ),
            "example_selected_sensors": (
                selected_lists[0]
                if selected_lists
                else ""
            ),
            "macro_f1_mean": np.mean(
                fold_macro_f1
            ),
            "macro_f1_std": np.std(
                fold_macro_f1,
                ddof=1,
            ),
            "accuracy_mean": np.mean(
                fold_acc
            ),
            "accuracy_std": np.std(
                fold_acc,
                ddof=1,
            ),
            "mcc_mean": np.mean(
                fold_mcc
            ),
            "mcc_std": np.std(
                fold_mcc,
                ddof=1,
            ),
            "mean_train_time_sec": np.mean(
                fit_times
            ),
            "mean_inference_time_sec": np.mean(
                pred_times
            ),
            "total_features": k,
        })

        logger.info(
            f"K={k}: "
            f"Macro F1={np.mean(fold_macro_f1):.4f} "
            f"+ or - {np.std(fold_macro_f1, ddof=1):.4f}"
        )

    ablation_df = pd.DataFrame(
        rows
    )

    ablation_df.to_csv(
        RESULTS_DIR / "ablation_results.csv",
        index=False,
    )

    logger.info(
        "Saved results/ablation_results.csv"
    )

    return ablation_df
