"""
Evaluation module: given CV results, persists the artifacts tied to the
single empirically-best model -- classification report, OOF predictions
(for the confusion matrix figure), and the final refit model + params.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef

from src.config import RESULTS_DIR, MODELS_DIR, ENC_TO_ORIG, ORIG_TO_MEANING, SEED, get_logger
from src.model_training import build_model_registry, build_pipeline_for

logger = get_logger("evaluation")


def save_oof_and_report(df, best_model_name: str, best_info: dict, y_encoded: np.ndarray,
                         y_original: np.ndarray):
    oof_pred_enc = best_info["oof_pred"]
    oof_pred_orig = np.array([ENC_TO_ORIG[v] for v in oof_pred_enc])

    oof_df = pd.DataFrame({
        "row_index": np.arange(len(y_original)),
        "Date": df["Date"].values,
        "Time": df["Time"].values,
        "y_true_original": y_original,
        "y_true_meaning": [ORIG_TO_MEANING[v] for v in y_original],
        "y_pred_original": oof_pred_orig,
        "y_pred_meaning": [ORIG_TO_MEANING[v] for v in oof_pred_orig],
        "correct": (oof_pred_orig == y_original),
    })
    oof_df.to_csv(RESULTS_DIR / "oof_predictions_best.csv", index=False)

    target_names = [f"{ENC_TO_ORIG[c]} ({ORIG_TO_MEANING[ENC_TO_ORIG[c]]})"
                     for c in sorted(np.unique(y_encoded))]
    report_dict = classification_report(
        y_encoded, oof_pred_enc, target_names=target_names, digits=4, zero_division=0,
        output_dict=True)

    mcc = matthews_corrcoef(y_encoded, oof_pred_enc)

    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(
        columns={"index": "class_or_average"})
    report_df.insert(0, "best_model", best_model_name)
    report_df.insert(1, "best_hyperparameters", str(best_info["params"]))
    report_df.insert(2, "n_oof_rows", len(y_original))
    report_df.insert(3, "overall_mcc", round(mcc, 6))
    report_df.to_csv(RESULTS_DIR / "classification_report_best.csv", index=False)

    logger.info(f"Saved OOF predictions + classification report (CSV) for best model {best_model_name}")
    return oof_df


def fit_and_save_final_model(X: pd.DataFrame, y_encoded: np.ndarray, best_model_name: str,
                              best_info: dict, n_jobs: int = -1):
    """Refit the winning model+config on the FULL dataset (documented protocol:
    CV estimates generalization; the persisted model is trained on all available
    data for deployment, as is standard practice once model selection is frozen)."""
    # Primary experiment safety check: final model must receive PIR sensors only.
    non_pir = [c for c in X.columns if not str(c).startswith("PIR_")]
    if non_pir:
        raise ValueError(
            "Final primary model received non PIR features: "
            f"{non_pir}. The primary experiment is PIR only."
        )

    registry = build_model_registry(n_jobs=n_jobs)
    spec = registry[best_model_name]
    params = best_info["params"]
    estimator = spec["estimator_fn"](**params)
    pipe = build_pipeline_for(best_model_name, estimator)
    pipe.fit(X, y_encoded)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(pipe, model_path)

    config_path = MODELS_DIR / "best_model_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name": best_model_name,
            "params": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in params.items()},
            "seed": SEED,
            "feature_policy": "PIR sensors only",
            "refit_protocol": "Refit on the full PIR-only dataset after model/hyperparameter "
                               "selection was frozen via 5-fold StratifiedGroupKFold CV "
                               "(temporal-group-aware). CV metrics in results/model_cv_summary.csv "
                               "are the generalization estimate; this artifact is the deployment model.",
            "feature_columns": list(X.columns),
        }, f, indent=2)

    logger.info(f"Saved final refit model -> {model_path}")
    return pipe, model_path


def save_all_model_params(cv_summary_df: pd.DataFrame, best_per_model_raw: dict):
    """Save results/final_model_parameters.json with each model's chosen params."""
    out = {}
    for _, row in cv_summary_df.iterrows():
        out[row["model"]] = {
            "best_config_id": int(row["best_config_id"]),
            "best_params": row["best_params"],
            "mean_macro_f1": row["macro_f1_mean"],
            "mean_accuracy": row["accuracy_mean"],
            "mean_mcc": row["mcc_mean"],
        }
    with open(RESULTS_DIR / "final_model_parameters.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Saved results/final_model_parameters.json")


def _count_params(estimator) -> int | None:
    """Best-effort parameter/complexity count for reporting purposes."""
    try:
        name = type(estimator).__name__
        if hasattr(estimator, "coef_"):
            return int(np.size(estimator.coef_)) + int(np.size(getattr(estimator, "intercept_", 0)))
        if hasattr(estimator, "n_features_in_") and hasattr(estimator, "coefs_"):  # MLP
            return int(sum(w.size for w in estimator.coefs_) + sum(b.size for b in estimator.intercepts_))
        if hasattr(estimator, "estimators_"):  # RF / ExtraTrees
            return int(sum(t.tree_.node_count for t in estimator.estimators_))
        if hasattr(estimator, "tree_"):  # DecisionTree
            return int(estimator.tree_.node_count)
        if hasattr(estimator, "get_booster"):  # XGBoost
            return int(estimator.get_booster().trees_to_dataframe().shape[0])
        if name == "CatBoostClassifier":
            return int(estimator.tree_count_)
        if hasattr(estimator, "n_iter_") and hasattr(estimator, "_baseline_prediction"):  # HistGB
            return sum(len(p.nodes) for pred in estimator._predictors for p in pred)
    except Exception:
        pass
    return None


def compute_computational_efficiency(X: pd.DataFrame, y_encoded: np.ndarray,
                                      cv_summary_df: pd.DataFrame, n_jobs: int = -1):
    """
    Refits each model's best config ONCE on the full dataset to measure:
    training time, inference time (on the full X), on-disk model size, and
    parameter/complexity count -- alongside the CV Macro F1 / Accuracy already
    computed. Saves results/computational_efficiency.csv.
    """
    registry = build_model_registry(n_jobs=n_jobs)
    rows = []

    for _, row in cv_summary_df.iterrows():
        model_name = row["model"]
        if model_name not in registry:
            continue
        params = ast.literal_eval(row["best_params"])
        spec = registry[model_name]
        estimator = spec["estimator_fn"](**params)
        pipe = build_pipeline_for(model_name, estimator)

        t0 = time.perf_counter()
        pipe.fit(X, y_encoded)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipe.predict(X)
        infer_time_total = time.perf_counter() - t0
        infer_time_per_sample_ms = 1000 * infer_time_total / len(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
            joblib.dump(pipe, tf.name)
            size_kb = os.path.getsize(tf.name) / 1024.0
        os.unlink(tf.name)

        n_params = _count_params(pipe.named_steps["model"])

        rows.append({
            "model": model_name,
            "train_time_sec_full_data": train_time,
            "inference_time_ms_per_sample": infer_time_per_sample_ms,
            "model_size_kb": round(size_kb, 2),
            "n_parameters_or_nodes": n_params,
            "macro_f1_cv_mean": row["macro_f1_mean"],
            "accuracy_cv_mean": row["accuracy_mean"],
            "mcc_cv_mean": row["mcc_mean"],
        })
        logger.info(f"Efficiency profiled: {model_name} "
                    f"train={train_time:.3f}s size={size_kb:.1f}KB")

    eff_df = pd.DataFrame(rows).sort_values("macro_f1_cv_mean", ascending=False).reset_index(drop=True)
    eff_df.to_csv(RESULTS_DIR / "computational_efficiency.csv", index=False)
    logger.info("Saved results/computational_efficiency.csv")
    return eff_df
