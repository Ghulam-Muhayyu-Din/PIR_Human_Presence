#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    PROJECT_ROOT,
    CANONICAL_CSV,
    RESULTS_DIR,
    MODELS_DIR,
    set_global_seed,
    get_logger,
)

logger = get_logger("run_all")


def announce(message: str) -> None:
    print(message, flush=True)
    logger.info(message)


def step(name):
    def deco(fn):
        def wrapper(*args, **kwargs):

            announce("=" * 80)
            announce(f"START: {name}")
            announce("=" * 80)

            t0 = time.perf_counter()

            result = fn(*args, **kwargs)

            elapsed = time.perf_counter() - t0

            announce(
                f"COMPLETED: {name} | "
                f"Elapsed time: {elapsed:.2f} seconds"
            )

            return result

        return wrapper

    return deco


@step("0. Validate single dataset")
def validate_dataset():

    if not CANONICAL_CSV.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n"
            f"{CANONICAL_CSV}\n\n"
            f"The pipeline requires exactly one dataset:\n"
            f"data/pirvision_office_dataset.csv"
        )

    if not CANONICAL_CSV.is_file():
        raise FileNotFoundError(
            f"Dataset path exists but is not a file:\n"
            f"{CANONICAL_CSV}"
        )

    file_size = CANONICAL_CSV.stat().st_size

    if file_size <= 0:
        raise ValueError(
            f"Dataset is empty:\n"
            f"{CANONICAL_CSV}"
        )

    size_mb = file_size / (1024 * 1024)

    announce(
        f"Dataset verified successfully: {CANONICAL_CSV}"
    )

    announce(
        f"Dataset size: {size_mb:.3f} MB"
    )


@step("1. Dataset audit")
def audit_step():

    from src.data_inspection import run_audit

    result = run_audit()

    announce("Dataset audit completed.")

    return result


@step("2. Feature identification")
def feature_id_step():

    from src.feature_identification import identify_features

    result = identify_features()

    announce("Feature identification completed.")

    return result


@step("3. Preprocessing and feature matrix")
def preprocessing_step():

    from src.preprocessing import load_dataset

    result = load_dataset()

    announce(
        f"Feature matrix created: "
        f"{result.X.shape[0]:,} rows × "
        f"{result.X.shape[1]:,} features"
    )

    announce(
        f"Encoded classes: "
        f"{sorted(set(result.y_encoded.tolist()))}"
    )

    announce(
        f"Original classes: "
        f"{sorted(set(result.y_original.tolist()))}"
    )

    return result


@step("4. Cross validation across model zoo")
def cv_step(ds):

    from src.cross_validation import run_cross_validation

    announce(
        "Beginning model training and validation."
    )

    return run_cross_validation(
        ds.X,
        ds.y_encoded,
        ds.y_original,
        ds.groups,
    )


@step("5. Evaluation and out of fold predictions")
def eval_report_step(
    ds,
    best_model_name,
    best_info,
):

    from src.evaluation import save_oof_and_report

    announce(
        f"Generating out of fold evaluation for: "
        f"{best_model_name}"
    )

    return save_oof_and_report(
        ds.df,
        best_model_name,
        best_info,
        ds.y_encoded,
        ds.y_original,
    )


@step("6. Refit and save final model")
def refit_step(
    ds,
    best_model_name,
    best_info,
):

    from src.evaluation import fit_and_save_final_model

    announce(
        f"Refitting final model: {best_model_name}"
    )

    pipe, model_path = fit_and_save_final_model(
        ds.X,
        ds.y_encoded,
        best_model_name,
        best_info,
    )

    announce(
        f"Final model saved to: {model_path}"
    )

    return pipe, model_path


@step("7. Save model parameters")
def save_params_step(
    cv_summary_df,
    best_per_model_raw=None,
):

    from src.evaluation import save_all_model_params

    save_all_model_params(
        cv_summary_df,
        best_per_model_raw,
    )

    announce(
        "Final model parameter file saved."
    )


@step("8. Explainability")
def explain_step(
    pipe,
    ds,
    best_model_name,
):

    from src.explainability import run_explainability

    announce(
        f"Running explainability for: "
        f"{best_model_name}"
    )

    importance_df, method_used = run_explainability(
        pipe,
        ds.X,
        ds.y_encoded,
        best_model_name,
    )

    announce(
        f"Explainability completed using: "
        f"{method_used}"
    )

    return importance_df, method_used


@step("9. PIR sensor ablation")
def ablation_step(
    ds,
    importance_df,
    best_model_name,
    best_params,
):

    from src.sensor_ablation import run_sensor_ablation

    announce(
        "Testing reduced PIR sensor configurations."
    )

    result = run_sensor_ablation(
        ds.X,
        ds.y_encoded,
        ds.groups,
        ds.pir_cols,
        importance_df,
        best_model_name,
        best_params,
    )

    announce(
        "Sensor ablation completed."
    )

    return result


@step("10. Computational efficiency")
def efficiency_step(
    ds,
    cv_summary_df,
):

    from src.evaluation import compute_computational_efficiency

    announce(
        "Calculating training and prediction efficiency."
    )

    result = compute_computational_efficiency(
        ds.X,
        ds.y_encoded,
        cv_summary_df,
    )

    announce(
        "Computational efficiency analysis completed."
    )

    return result


@step("11. Temporal analysis")
def temporal_step(
    ds,
    best_model_name,
    best_params,
):

    from src.temporal_analysis import run_temporal_analysis

    announce(
        "Running temporal robustness analysis."
    )

    result = run_temporal_analysis(
        ds.df,
        ds.X,
        ds.y_encoded,
        ds.y_original,
        best_model_name,
        best_params,
    )

    announce(
        "Temporal analysis completed."
    )

    return result


@step("12. Figures and figure quality control")
def figures_step(
    ds,
    cv_summary_df,
    oof_df,
    ablation_df,
    eff_df,
    best_model_name,
    temporal_result,
):

    from src import figures
    from src.figure_qc import run_figure_qc

    announce(
        "Generating publication quality figures."
    )

    figures.figure_01_workflow()

    figures.figure_02_class_distribution(
        ds.y_original
    )

    figures.figure_03_model_comparison(
        cv_summary_df
    )

    figures.figure_04_confusion_matrix(
        oof_df[
            "y_true_original"
        ].to_numpy(),
        oof_df[
            "y_pred_original"
        ].to_numpy(),
        best_model_name,
    )

    figures.figure_06_pir_sensor_reduction(
        ablation_df
    )

    figures.figure_07_performance_vs_cost(
        eff_df
    )

    figures.figure_08_correlation_heatmap(
        ds.X,
        ds.pir_cols,
    )

    figures.figure_09_per_class_performance(
        cv_summary_df,
        best_model_name,
    )

    figures.figure_10_temporal_performance(
        ds.df,
        temporal_result,
    )

    announce(
        "All figure functions executed."
    )

    qc_result = run_figure_qc()

    announce(
        "Figure quality control completed."
    )

    return qc_result


@step("13. Scientific results summary")
def report_step():

    from src.report_generation import generate_report

    result = generate_report()

    announce(
        "Scientific results summary generated."
    )

    return result


def main():

    t_start = time.perf_counter()

    print()
    print("=" * 80)
    print(
        "PIR HUMAN PRESENCE DETECTION RESEARCH PIPELINE"
    )
    print("=" * 80)
    print(
        "Pipeline execution started."
    )
    print(
        f"Python executable: {sys.executable}"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Dataset: {CANONICAL_CSV}"
    )
    print("=" * 80)
    print()

    set_global_seed(42)

    validate_dataset()

    audit_step()

    feature_id_step()

    ds = preprocessing_step()

    (
        fold_results_df,
        cv_summary_df,
        best_model_name,
        best_info,
    ) = cv_step(ds)

    oof_df = eval_report_step(
        ds,
        best_model_name,
        best_info,
    )

    pipe, model_path = refit_step(
        ds,
        best_model_name,
        best_info,
    )

    save_params_step(
        cv_summary_df
    )

    importance_df, method_used = explain_step(
        pipe,
        ds,
        best_model_name,
    )

    ablation_df = ablation_step(
        ds,
        importance_df,
        best_model_name,
        best_info["params"],
    )

    eff_df = efficiency_step(
        ds,
        cv_summary_df,
    )

    temporal_result = temporal_step(
        ds,
        best_model_name,
        best_info["params"],
    )

    figures_step(
        ds,
        cv_summary_df,
        oof_df,
        ablation_df,
        eff_df,
        best_model_name,
        temporal_result,
    )

    report_step()

    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 80)
    print(
        "PIPELINE COMPLETE"
    )
    print("=" * 80)

    print(
        f"Total execution time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Cross validation Macro F1: "
        f"{best_info['mean_macro_f1']:.6f}"
    )

    print(
        f"Explainability: "
        f"{method_used}"
    )

    print(
        f"Results directory: "
        f"{RESULTS_DIR}"
    )

    print(
        f"Models directory: "
        f"{MODELS_DIR}"
    )

    print(
        f"Figures directory: "
        f"{PROJECT_ROOT / 'figures'}"
    )

    print("=" * 80)
    print()


if __name__ == "__main__":

    try:

        main()

    except Exception:

        print()
        print("=" * 80)
        print("PIPELINE FAILED")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)

        logger.error(
            "PIPELINE FAILED:\n"
            + traceback.format_exc()
        )

        raise