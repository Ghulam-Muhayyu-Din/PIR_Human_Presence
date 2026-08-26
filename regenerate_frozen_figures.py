from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import figures
from src.figure_qc import run_figure_qc

DATA = ROOT / "data" / "pirvision_office_dataset.csv"
RESULTS = ROOT / "results"


def main() -> None:
    print("Generating final figures from frozen experiment outputs.", flush=True)
    print("No model training will be performed.", flush=True)

    df = pd.read_csv(DATA)
    pir_cols = [c for c in df.columns if str(c).startswith("PIR_")]
    X = df[pir_cols].copy()
    y_original = df["Label"].to_numpy()

    cv_summary = pd.read_csv(
        RESULTS / "model_cv_summary.csv"
    )
    oof = pd.read_csv(
        RESULTS / "oof_predictions_best.csv"
    )
    ablation = pd.read_csv(
        RESULTS / "ablation_results.csv"
    )
    efficiency = pd.read_csv(
        RESULTS / "computational_efficiency.csv"
    )
    temporal = pd.read_csv(
        RESULTS / "temporal_analysis.csv"
    )

    best_model = str(cv_summary.iloc[0]["model"])

    # Reconstruct the exact final 15 percent chronological window used by
    # temporal_analysis.py. A holdout date can be only partially included.
    df_sorted = df.copy()
    df_sorted["_Timestamp"] = pd.to_datetime(
        df_sorted["Date"].astype(str) + " " + df_sorted["Time"].astype(str)
    )
    df_sorted = df_sorted.sort_values("_Timestamp").reset_index(drop=True)
    split_at = int(len(df_sorted) * 0.85)
    holdout_rows = (
        df_sorted.iloc[split_at:]["Date"]
        .value_counts()
        .to_dict()
    )

    temporal_result = {
        "holdout_row_counts_by_date": {
            str(k): int(v) for k, v in holdout_rows.items()
        }
    }

    figures.figure_01_workflow()
    figures.figure_02_class_distribution(y_original)
    figures.figure_03_model_comparison(cv_summary)
    figures.figure_04_confusion_matrix(
        oof["y_true_original"].to_numpy(),
        oof["y_pred_original"].to_numpy(),
        best_model,
    )
    figures.figure_06_pir_sensor_reduction(ablation)
    figures.figure_07_performance_vs_cost(efficiency)
    figures.figure_08_correlation_heatmap(X, pir_cols)
    figures.figure_09_per_class_performance(
        cv_summary,
        best_model,
    )
    figures.figure_10_temporal_performance(
        df,
        temporal_result,
    )

    qc = run_figure_qc()

    if not (qc["status"] == "OK").all():
        raise RuntimeError(
            "Final figure quality control did not pass for every figure.\n"
            + qc.to_string(index=False)
        )

    print("Final figure generation and QC completed successfully.", flush=True)


if __name__ == "__main__":
    main()
