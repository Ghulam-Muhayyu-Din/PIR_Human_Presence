"""
Temporal robustness analysis (brief section 5 continuation + section 15).

1. Inspects per-date class distribution (already saved by data_inspection.py
   to results/per_date_class_distribution.csv) to decide whether a strict
   final-chronological holdout is valid (i.e. contains >=2 classes).
2. If valid, reports it as a genuine chronological holdout benchmark.
   If NOT valid (holdout date(s) contain only one class), it is clearly
   labeled a "temporal stress analysis" -- NOT a 3-class benchmark -- and
   documented as a limitation, with actual class counts shown.
3. Also reports per-temporal-group (StratifiedGroupKFold fold) performance
   already computed in cross_validation.py, viewed through a date lens.

Outputs:
    results/temporal_analysis.csv   (tidy, machine-readable: per-date class
                                      counts + holdout evaluation metrics)
    results/temporal_holdout_report.txt (companion human-readable narrative)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_fscore_support, classification_report,
)

from src.config import RESULTS_DIR, ORIG_TO_MEANING, SEED, get_logger
from src.model_training import build_pipeline_for, build_model_registry

logger = get_logger("temporal_analysis")

HOLDOUT_FRACTION = 0.15  # final chronological slice reserved for the stress test


def run_temporal_analysis(df: pd.DataFrame, X: pd.DataFrame, y_encoded: np.ndarray,
                           y_original: np.ndarray, best_model_name: str, best_params: dict,
                           n_jobs: int = -1):
    lines = []
    lines.append("=" * 78)
    lines.append("TEMPORAL ROBUSTNESS / CHRONOLOGICAL STRESS ANALYSIS")
    lines.append("=" * 78)
    lines.append("")

    per_date = pd.read_csv(RESULTS_DIR / "per_date_class_distribution.csv", index_col=0)
    lines.append("Per-date class distribution (rows=date, cols=Label):")
    lines.append(per_date.to_string())
    lines.append("")

    n = len(df)
    split_at = int(n * (1 - HOLDOUT_FRACTION))
    # df is already sorted chronologically by preprocessing.load_dataset
    train_idx = np.arange(0, split_at)
    holdout_idx = np.arange(split_at, n)

    holdout_dates = sorted(df.iloc[holdout_idx]["Date"].unique().tolist())
    holdout_classes = sorted(pd.unique(y_original[holdout_idx]).tolist())
    # exact row counts, per date, that fall inside the chronological holdout
    # window (a date can be only PARTIALLY inside the holdout window -- the
    # figure needs the true count, not just "is this date in holdout_dates").
    holdout_row_counts_by_date = df.iloc[holdout_idx]["Date"].value_counts().to_dict()

    lines.append(f"Final-chronological holdout window: last {HOLDOUT_FRACTION*100:.0f}% of rows "
                 f"({len(holdout_idx)} rows), covering dates: {holdout_dates}")
    lines.append(f"Classes present in this holdout window: {holdout_classes}")
    lines.append("")

    is_valid_multiclass_holdout = len(holdout_classes) >= 2

    registry = build_model_registry(n_jobs=n_jobs)
    spec = registry[best_model_name]
    estimator = spec["estimator_fn"](**best_params)
    pipe = build_pipeline_for(best_model_name, estimator)
    pipe.fit(X.iloc[train_idx], y_encoded[train_idx])
    y_pred = np.asarray(pipe.predict(X.iloc[holdout_idx])).reshape(-1)
    y_true = y_encoded[holdout_idx]

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if is_valid_multiclass_holdout:
        lines.append("RESULT TYPE: Valid strict chronological holdout benchmark "
                      "(>=2 classes present).")
        bacc = balanced_accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        lines.append(f"  Accuracy           : {acc:.4f}")
        lines.append(f"  Balanced Accuracy  : {bacc:.4f}")
        lines.append(f"  Macro Precision    : {prec:.4f}")
        lines.append(f"  Macro Recall       : {rec:.4f}")
        lines.append(f"  Macro F1           : {f1:.4f}")
        lines.append(f"  MCC                : {mcc:.4f}")
    else:
        lines.append("RESULT TYPE: *** TEMPORAL STRESS ANALYSIS ONLY -- NOT a 3-class benchmark ***")
        lines.append("  The final-chronological holdout window contains only "
                      f"{len(holdout_classes)} class(es): {holdout_classes} "
                      f"({[ORIG_TO_MEANING.get(c,'?') for c in holdout_classes]}).")
        lines.append("  Reporting Macro F1/Precision/Recall here would be misleading (undefined "
                      "or degenerate for absent classes), so only single-class-appropriate "
                      "metrics are reported, and this is documented as a dataset LIMITATION: "
                      "the tail of the collection period does not exercise all activity states, "
                      "so a genuine end-of-timeline multiclass holdout is not currently possible "
                      "with this dataset.")
        lines.append(f"  Accuracy on holdout (single/degenerate class set) : {acc:.4f}")
        lines.append(f"  MCC on holdout                                    : {mcc:.4f}")
        vc = pd.Series(y_original[holdout_idx]).value_counts()
        lines.append(f"  Actual holdout class counts: {vc.to_dict()}")

    lines.append("")
    lines.append("Principal evaluation remains the 5-fold StratifiedGroupKFold CV "
                  "(results/model_cv_summary.csv), which IS a valid multiclass, "
                  "temporally-leakage-controlled benchmark across the full date range.")
    lines.append("=" * 78)

    with open(RESULTS_DIR / "temporal_holdout_report.txt", "w") as f:
        f.write("\n".join(lines))

    # --- tidy, machine-readable CSV (section: results/temporal_analysis.csv) ----
    csv_rows = []
    for date, row in per_date.iterrows():
        n_row = int(row.sum())
        csv_rows.append({
            "section": "per_date_class_distribution",
            "date": date,
            "n_rows": n_row,
            "count_label_0_vacancy": int(row.get("0", row.get(0, 0))),
            "count_label_1_stationary": int(row.get("1", row.get(1, 0))),
            "count_label_3_motion": int(row.get("3", row.get(3, 0))),
            "is_in_chronological_holdout": date in holdout_dates,
            "metric": None,
            "value": None,
        })
    holdout_metrics = {
        "holdout_fraction": HOLDOUT_FRACTION,
        "holdout_n_rows": len(holdout_idx),
        "holdout_dates": ";".join(holdout_dates),
        "holdout_classes_present": ";".join(str(c) for c in holdout_classes),
        "is_valid_multiclass_holdout": is_valid_multiclass_holdout,
        "holdout_accuracy": round(float(acc), 6),
        "holdout_mcc": round(float(mcc), 6),
    }
    if is_valid_multiclass_holdout:
        holdout_metrics.update({
            "holdout_balanced_accuracy": round(float(bacc), 6),
            "holdout_macro_precision": round(float(prec), 6),
            "holdout_macro_recall": round(float(rec), 6),
            "holdout_macro_f1": round(float(f1), 6),
        })
    for k, v in holdout_metrics.items():
        csv_rows.append({
            "section": "chronological_holdout_evaluation",
            "date": None, "n_rows": None,
            "count_label_0_vacancy": None, "count_label_1_stationary": None,
            "count_label_3_motion": None, "is_in_chronological_holdout": None,
            "metric": k, "value": v,
        })
    temporal_df = pd.DataFrame(csv_rows)
    temporal_df.to_csv(RESULTS_DIR / "temporal_analysis.csv", index=False)

    logger.info(f"Temporal analysis complete. Valid multiclass holdout: {is_valid_multiclass_holdout}")

    return {
        "is_valid_multiclass_holdout": is_valid_multiclass_holdout,
        "holdout_dates": holdout_dates,
        "holdout_classes": holdout_classes,
        "holdout_accuracy": acc,
        "holdout_mcc": mcc,
        "per_date": per_date,
        "holdout_row_counts_by_date": holdout_row_counts_by_date,
    }
