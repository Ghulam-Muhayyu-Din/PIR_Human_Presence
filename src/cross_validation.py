from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    matthews_corrcoef,
)

from src.config import (
    SEED,
    N_SPLITS,
    RESULTS_DIR,
    ENC_TO_ORIG,
    get_logger,
)

from src.model_training import (
    build_model_registry,
    build_pipeline_for,
)


logger = get_logger(
    "cross_validation"
)


def announce(message: str) -> None:

    print(message, flush=True)

    logger.info(message)


def _fold_metrics(
    y_true_enc,
    y_pred_enc,
    n_classes,
):

    acc = accuracy_score(
        y_true_enc,
        y_pred_enc,
    )

    bacc = balanced_accuracy_score(
        y_true_enc,
        y_pred_enc,
    )

    prec, rec, f1, _ = (
        precision_recall_fscore_support(
            y_true_enc,
            y_pred_enc,
            average="macro",
            zero_division=0,
        )
    )

    f1_weighted = f1_score(
        y_true_enc,
        y_pred_enc,
        average="weighted",
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true_enc,
        y_pred_enc,
    )

    (
        per_class_p,
        per_class_r,
        per_class_f1,
        per_class_support,
    ) = precision_recall_fscore_support(
        y_true_enc,
        y_pred_enc,
        labels=list(
            range(n_classes)
        ),
        average=None,
        zero_division=0,
    )

    out = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "weighted_f1": f1_weighted,
        "mcc": mcc,
    }

    for c in range(n_classes):

        orig = ENC_TO_ORIG[c]

        out[
            f"precision_class_{orig}"
        ] = per_class_p[c]

        out[
            f"recall_class_{orig}"
        ] = per_class_r[c]

        out[
            f"f1_class_{orig}"
        ] = per_class_f1[c]

        out[
            f"support_class_{orig}"
        ] = per_class_support[c]

    return out


def run_cross_validation(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    y_original: np.ndarray,
    groups: np.ndarray,
    n_jobs: int = -1,
    feature_subset: list | None = None,
    model_names: list | None = None,
):

    """
    Principal group aware cross validation.

    Every model configuration is trained again on every fold.

    No cached model is used.

    The output contains fold level results, model level
    summaries, and the empirically selected model.
    """

    total_start = time.perf_counter()

    announce("")
    announce("=" * 80)
    announce("MODEL TRAINING AND CROSS VALIDATION STARTED")
    announce("=" * 80)

    if feature_subset is not None:

        missing = [
            c
            for c in feature_subset
            if c not in X.columns
        ]

        if missing:

            raise ValueError(
                f"Requested feature subset contains "
                f"unknown columns: {missing}"
            )

        X = X[
            feature_subset
        ].copy()

        announce(
            f"Feature subset active: "
            f"{X.shape[1]} features"
        )

    else:

        X = X.copy()

    y_encoded = np.asarray(
        y_encoded
    )

    y_original = np.asarray(
        y_original
    )

    groups = np.asarray(
        groups
    )

    if len(X) != len(y_encoded):

        raise ValueError(
            "X and y_encoded have different lengths."
        )

    if len(X) != len(groups):

        raise ValueError(
            "X and groups have different lengths."
        )

    if len(y_original) != len(y_encoded):

        raise ValueError(
            "y_original and y_encoded have different lengths."
        )

    n_classes = len(
        np.unique(y_encoded)
    )

    announce(
        f"Observations: {len(X):,}"
    )

    announce(
        f"Features: {X.shape[1]:,}"
    )

    announce(
        f"Classes: {n_classes}"
    )

    announce(
        f"Cross validation folds: {N_SPLITS}"
    )

    announce(
        "Validation strategy: "
        "StratifiedGroupKFold"
    )

    announce(
        f"Unique temporal groups: "
        f"{len(np.unique(groups)):,}"
    )

    registry = (
        build_model_registry(
            n_jobs=n_jobs
        )
    )

    if model_names is not None:

        registry = {
            k: v
            for k, v in registry.items()
            if k in model_names
        }

    if not registry:

        raise ValueError(
            "No models are available for cross validation."
        )

    announce(
        f"Models to evaluate: "
        f"{len(registry)}"
    )

    announce(
        "Model list: "
        + ", ".join(
            registry.keys()
        )
    )

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

    if len(fold_splits) != N_SPLITS:

        raise RuntimeError(
            "The number of generated folds "
            "does not match the configured number."
        )

    for fold_number, (
        tr_idx,
        va_idx,
    ) in enumerate(
        fold_splits,
        start=1,
    ):

        train_groups = set(
            groups[tr_idx]
        )

        valid_groups = set(
            groups[va_idx]
        )

        overlap = (
            train_groups
            .intersection(
                valid_groups
            )
        )

        if overlap:

            raise RuntimeError(
                f"Temporal group leakage detected "
                f"in fold {fold_number}."
            )

    fold_rows = []

    best_per_model = {}

    total_configurations = sum(
        len(
            spec["param_grid"]
        )
        for spec in registry.values()
    )

    announce(
        f"Total model configurations: "
        f"{total_configurations}"
    )

    global_config_counter = 0

    for model_name, spec in registry.items():

        config_count = len(
            spec["param_grid"]
        )

        announce("")
        announce("=" * 80)

        announce(
            f"MODEL: {model_name}"
        )

        announce(
            f"Configurations: {config_count}"
        )

        announce("=" * 80)

        for cfg_idx, params in enumerate(
            spec["param_grid"]
        ):

            global_config_counter += 1

            announce("")
            announce(
                f"CONFIGURATION "
                f"{cfg_idx + 1}/{config_count}"
            )

            announce(
                f"Overall configuration "
                f"{global_config_counter}/{total_configurations}"
            )

            announce(
                f"Model: {model_name}"
            )

            announce(
                f"Parameters: {params}"
            )

            oof_pred = np.full(
                len(y_encoded),
                -1,
                dtype=int,
            )

            fit_times = []
            pred_times = []
            fold_macro_f1s = []

            configuration_start = (
                time.perf_counter()
            )

            for fold_idx, (
                tr_idx,
                va_idx,
            ) in enumerate(
                fold_splits
            ):

                fold_number = (
                    fold_idx + 1
                )

                X_tr = X.iloc[
                    tr_idx
                ]

                X_va = X.iloc[
                    va_idx
                ]

                y_tr = y_encoded[
                    tr_idx
                ]

                y_va = y_encoded[
                    va_idx
                ]

                train_classes = set(
                    np.unique(y_tr)
                )

                validation_classes = set(
                    np.unique(y_va)
                )

                announce(
                    ""
                )

                announce(
                    f"TRAINING "
                    f"{model_name} "
                    f"| configuration "
                    f"{cfg_idx + 1}/{config_count} "
                    f"| fold "
                    f"{fold_number}/{N_SPLITS}"
                )

                announce(
                    f"Train rows: "
                    f"{len(tr_idx):,} | "
                    f"Validation rows: "
                    f"{len(va_idx):,}"
                )

                announce(
                    f"Training classes: "
                    f"{sorted(train_classes)}"
                )

                announce(
                    f"Validation classes: "
                    f"{sorted(validation_classes)}"
                )

                try:

                    estimator = (
                        spec[
                            "estimator_fn"
                        ](**params)
                    )

                except Exception as exc:

                    raise RuntimeError(
                        f"Could not create "
                        f"{model_name} "
                        f"configuration {cfg_idx}: "
                        f"{exc}"
                    ) from exc

                pipe = build_pipeline_for(
                    model_name,
                    estimator,
                )

                fit_start = (
                    time.perf_counter()
                )

                try:

                    pipe.fit(
                        X_tr,
                        y_tr,
                    )

                except Exception as exc:

                    raise RuntimeError(
                        f"Training failed for "
                        f"{model_name}, "
                        f"configuration {cfg_idx}, "
                        f"fold {fold_number}: "
                        f"{exc}"
                    ) from exc

                fit_time = (
                    time.perf_counter()
                    - fit_start
                )

                fit_times.append(
                    fit_time
                )

                announce(
                    f"Training finished in "
                    f"{fit_time:.3f} seconds"
                )

                predict_start = (
                    time.perf_counter()
                )

                try:

                    y_pred = np.asarray(
                        pipe.predict(
                            X_va
                        )
                    ).reshape(-1)

                except Exception as exc:

                    raise RuntimeError(
                        f"Prediction failed for "
                        f"{model_name}, "
                        f"configuration {cfg_idx}, "
                        f"fold {fold_number}: "
                        f"{exc}"
                    ) from exc

                pred_time = (
                    time.perf_counter()
                    - predict_start
                )

                pred_times.append(
                    pred_time
                )

                oof_pred[
                    va_idx
                ] = y_pred

                metrics = _fold_metrics(
                    y_va,
                    y_pred,
                    n_classes,
                )

                fold_macro_f1s.append(
                    metrics[
                        "macro_f1"
                    ]
                )

                announce(
                    f"Validation Macro F1: "
                    f"{metrics['macro_f1']:.6f}"
                )

                announce(
                    f"Validation Accuracy: "
                    f"{metrics['accuracy']:.6f}"
                )

                announce(
                    f"Validation MCC: "
                    f"{metrics['mcc']:.6f}"
                )

                row = {
                    "model": model_name,
                    "config_id": cfg_idx,
                    "params": str(params),
                    "fold": fold_number,
                    "n_train": len(tr_idx),
                    "n_val": len(va_idx),
                    "fit_time_sec": fit_time,
                    "predict_time_sec": pred_time,
                }

                row.update(
                    metrics
                )

                fold_rows.append(
                    row
                )

            if not fold_macro_f1s:

                raise RuntimeError(
                    f"No successful folds for "
                    f"{model_name}, "
                    f"configuration {cfg_idx}."
                )

            mean_macro_f1 = float(
                np.mean(
                    fold_macro_f1s
                )
            )

            std_macro_f1 = float(
                np.std(
                    fold_macro_f1s,
                    ddof=1,
                )
                if len(
                    fold_macro_f1s
                ) > 1
                else 0.0
            )

            mean_fit_time = float(
                np.mean(
                    fit_times
                )
            )

            mean_pred_time = float(
                np.mean(
                    pred_times
                )
            )

            configuration_elapsed = (
                time.perf_counter()
                - configuration_start
            )

            announce("")
            announce(
                f"COMPLETED CONFIGURATION: "
                f"{model_name} "
                f"cfg {cfg_idx}"
            )

            announce(
                f"Macro F1: "
                f"{mean_macro_f1:.6f} "
                f"+ or minus "
                f"{std_macro_f1:.6f}"
            )

            announce(
                f"Mean training time: "
                f"{mean_fit_time:.3f} seconds"
            )

            announce(
                f"Mean prediction time: "
                f"{mean_pred_time:.3f} seconds"
            )

            announce(
                f"Configuration elapsed time: "
                f"{configuration_elapsed:.2f} seconds"
            )

            current_best = (
                best_per_model.get(
                    model_name
                )
            )

            if (
                current_best is None
                or mean_macro_f1
                > current_best[
                    "mean_macro_f1"
                ]
            ):

                best_per_model[
                    model_name
                ] = {
                    "config_id": cfg_idx,
                    "params": params,
                    "mean_macro_f1": (
                        mean_macro_f1
                    ),
                    "std_macro_f1": (
                        std_macro_f1
                    ),
                    "oof_pred": (
                        oof_pred.copy()
                    ),
                    "mean_fit_time": (
                        mean_fit_time
                    ),
                    "mean_pred_time": (
                        mean_pred_time
                    ),
                }

                announce(
                    f"NEW BEST CONFIGURATION "
                    f"FOR {model_name}: "
                    f"cfg {cfg_idx} "
                    f"| Macro F1 = "
                    f"{mean_macro_f1:.6f}"
                )

    fold_results_df = pd.DataFrame(
        fold_rows
    )

    if fold_results_df.empty:

        raise RuntimeError(
            "Cross validation produced no results."
        )

    fold_results_df.to_csv(
        RESULTS_DIR
        / "model_fold_results.csv",
        index=False,
    )

    summary_rows = []

    for model_name, info in (
        best_per_model.items()
    ):

        sub = fold_results_df[
            (
                fold_results_df[
                    "model"
                ]
                == model_name
            )
            &
            (
                fold_results_df[
                    "config_id"
                ]
                == info[
                    "config_id"
                ]
            )
        ]

        metric_cols = [
            c
            for c in sub.columns
            if c not in (
                "model",
                "config_id",
                "params",
                "fold",
                "n_train",
                "n_val",
                "fit_time_sec",
                "predict_time_sec",
            )
        ]

        aggregate_mean = {
            f"{c}_mean": sub[
                c
            ].mean()
            for c in metric_cols
        }

        aggregate_std = {
            f"{c}_std": sub[
                c
            ].std(
                ddof=1
            )
            for c in metric_cols
        }

        summary_rows.append(
            {
                "model": model_name,
                "best_config_id": (
                    info[
                        "config_id"
                    ]
                ),
                "best_params": str(
                    info[
                        "params"
                    ]
                ),
                "mean_fit_time_sec": (
                    info[
                        "mean_fit_time"
                    ]
                ),
                "mean_predict_time_sec": (
                    info[
                        "mean_pred_time"
                    ]
                ),
                **aggregate_mean,
                **aggregate_std,
            }
        )

    cv_summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            by=[
                "macro_f1_mean",
                "mcc_mean",
                "accuracy_mean",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    cv_summary_df.to_csv(
        RESULTS_DIR
        / "model_cv_summary.csv",
        index=False,
    )

    best_model_name = (
        cv_summary_df.iloc[
            0
        ]["model"]
    )

    best_info = (
        best_per_model[
            best_model_name
        ]
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    announce("")
    announce("=" * 80)
    announce(
        "MODEL TRAINING AND CROSS VALIDATION COMPLETE"
    )
    announce("=" * 80)

    announce(
        f"Best model: "
        f"{best_model_name}"
    )

    announce(
        f"Best Macro F1: "
        f"{best_info['mean_macro_f1']:.6f}"
    )

    announce(
        f"Best configuration: "
        f"{best_info['config_id']}"
    )

    announce(
        f"Total CV time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    announce("")
    announce(
        "FINAL MODEL RANKING"
    )

    ranking_columns = [
        "model",
        "macro_f1_mean",
        "accuracy_mean",
        "mcc_mean",
        "mean_fit_time_sec",
    ]

    print(
        cv_summary_df[
            ranking_columns
        ].to_string(
            index=False
        ),
        flush=True,
    )

    logger.info(
        f"Empirical best model: "
        f"{best_model_name} | "
        f"Macro F1 = "
        f"{best_info['mean_macro_f1']:.6f}"
    )

    return (
        fold_results_df,
        cv_summary_df,
        best_model_name,
        best_info,
    )