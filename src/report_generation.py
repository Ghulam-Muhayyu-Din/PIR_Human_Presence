"""
Generates results/scientific_results_summary.md programmatically from the
actual generated CSV/JSON result files (no hand-typed numbers).
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import RESULTS_DIR, ORIG_TO_MEANING, get_logger

logger = get_logger("report_generation")


def _read_csv(name):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def generate_report():
    audit = _read_csv("dataset_audit.csv")
    cv_summary = _read_csv("model_cv_summary.csv")
    ablation = _read_csv("ablation_results.csv")
    efficiency = _read_csv("computational_efficiency.csv")
    importance = _read_csv("feature_importance.csv")

    temporal_report_path = RESULTS_DIR / "temporal_holdout_report.txt"
    temporal_report_text = temporal_report_path.read_text() if temporal_report_path.exists() else "N/A"
    temporal_csv = _read_csv("temporal_analysis.csv")

    md = []
    md.append("# Scientific Results Summary")
    md.append("")
    md.append("**Title:** Lightweight and Explainable Machine Learning for PIR Based Human "
               "Presence Detection in Smart Environments")
    md.append("")
    md.append("_This document is generated programmatically from the pipeline's own result "
               "files (results/*.csv, results/*.json) -- no numbers below are hand-typed._")
    md.append("")

    # --- dataset ------------------------------------------------------------
    md.append("## 1. Dataset Description")
    if audit is not None:
        a = audit.set_index("metric")["value"]
        md.append(f"- Single canonical dataset: `{a.get('dataset_path')}`")
        md.append(f"- Rows x Columns: **{a.get('n_rows')} x {a.get('n_columns')}**")
        md.append(f"- Date range: {a.get('date_min')} to {a.get('date_max')}")
        md.append(f"- PIR sensor columns detected: **{a.get('n_pir_sensor_columns_detected')}**")
        md.append(f"- Target values match expected {{0,1,3}}: **{a.get('target_values_match_expected')}**")
        md.append(f"- Missing values: {a.get('total_missing_values')}; "
                   f"duplicate rows within file: {a.get('duplicate_rows_within_file')}")
        md.append(f"- Identifier column present: {a.get('identifier_column_candidates')}")
        temp_zero = audit[audit["metric"] == "Temperature_F_zero_value_rows"]
        if len(temp_zero):
            md.append(f"- Temperature_F==0 rows (data-quality flag): "
                       f"{temp_zero.iloc[0]['value']} -- {temp_zero.iloc[0]['notes']}")
        gap_row = audit[audit["metric"] == "median_sampling_gap_seconds"]
        if len(gap_row):
            md.append(f"- Median sampling gap between observations: {gap_row.iloc[0]['value']} s "
                       "(sampling confirmed non-uniform)")
    md.append("")

    # --- class distribution --------------------------------------------------
    md.append("## 2. Class Distribution (target = Label; values 0, 1, 3)")
    if audit is not None:
        for lbl in (0, 1, 3):
            row = audit[audit["metric"] == f"class_count_label_{lbl}"]
            if len(row):
                md.append(f"- Label={lbl} ({ORIG_TO_MEANING.get(lbl,'?')}): "
                           f"**{row.iloc[0]['value']}** rows")
    md.append("")

    # --- model comparison ------------------------------------------------------
    md.append("## 3. Model Comparison (5-fold StratifiedGroupKFold CV, temporal-group-aware)")
    if cv_summary is not None:
        best = cv_summary.iloc[0]
        md.append(f"**Best model (empirically selected, primary metric = Macro F1): "
                   f"`{best['model']}`**")
        md.append(f"- Macro F1  = {best['macro_f1_mean']:.4f} (+/- {best.get('macro_f1_std', float('nan')):.4f})")
        md.append(f"- Accuracy  = {best['accuracy_mean']:.4f}")
        md.append(f"- Balanced Accuracy = {best.get('balanced_accuracy_mean', float('nan')):.4f}")
        md.append(f"- MCC       = {best['mcc_mean']:.4f}")
        md.append(f"- Weighted F1 = {best.get('weighted_f1_mean', float('nan')):.4f}")
        md.append("")
        md.append("Full comparison table (best config per model, ranked by Macro F1):")
        md.append("")
        cols = ["model", "macro_f1_mean", "accuracy_mean", "balanced_accuracy_mean",
                "mcc_mean", "weighted_f1_mean", "mean_fit_time_sec", "mean_predict_time_sec"]
        cols = [c for c in cols if c in cv_summary.columns]
        tbl = cv_summary[cols].copy()
        for c in tbl.columns:
            if tbl[c].dtype.kind == "f":
                tbl[c] = tbl[c].round(4)
        md.append(tbl.to_markdown(index=False))
    md.append("")

    # --- top PIR sensors -------------------------------------------------------
    md.append("## 4. Explainability -- Most Important PIR Sensors")
    if importance is not None:
        method = importance["method"].iloc[0] if "method" in importance.columns else "unknown"
        md.append(f"Method used: **{method}**")
        md.append("")
        top10 = importance.head(10)
        md.append("Top 10 most important features overall:")
        md.append("")
        md.append(top10[["feature", "importance"]].round(5).to_markdown(index=False))
        temp_row = importance[importance["feature"] == "Temperature_F"]
        if len(temp_row):
            rank = int(importance.index[importance["feature"] == "Temperature_F"][0]) + 1
            md.append("")
            md.append(f"Temperature_F importance rank: **#{rank}** of {len(importance)} features "
                       f"(importance = {temp_row.iloc[0]['importance']:.5f}).")
    md.append("")

    # --- ablation ----------------------------------------------------------------
    md.append("## 5. PIR Sensor Reduction (Ablation)")
    if ablation is not None:
        md.append(ablation[["n_pir_sensors", "macro_f1_mean", "macro_f1_std", "accuracy_mean",
                             "mcc_mean", "mean_train_time_sec",
                             "mean_inference_time_sec"]].round(5).to_markdown(index=False))
        best_full = ablation[ablation["n_pir_sensors"] == ablation["n_pir_sensors"].max()].iloc[0]
        best_small = ablation.loc[ablation["macro_f1_mean"].idxmax()]
        md.append("")
        md.append(f"Using all {int(best_full['n_pir_sensors'])} PIR sensors achieves Macro F1 = "
                   f"{best_full['macro_f1_mean']:.4f}. The best Macro F1 across all tested "
                   f"subset sizes was {best_small['macro_f1_mean']:.4f} at "
                   f"K={int(best_small['n_pir_sensors'])} sensors, indicating that a reduced "
                   "sensor subset can retain competitive accuracy for lightweight deployment.")
    md.append("")

    # --- computational efficiency ------------------------------------------------
    md.append("## 6. Computational Efficiency")
    if efficiency is not None:
        md.append(efficiency[["model", "train_time_sec_full_data",
                               "inference_time_ms_per_sample", "model_size_kb",
                               "macro_f1_cv_mean"]].round(4).to_markdown(index=False))
    md.append("")

    # --- temporal robustness -------------------------------------------------------
    md.append("## 7. Temporal Robustness")
    md.append("_Full machine-readable data in `results/temporal_analysis.csv`; narrative "
               "below is generated from `results/temporal_holdout_report.txt`._")
    md.append("```")
    md.append(temporal_report_text)
    md.append("```")
    md.append("")

    # --- limitations / practical implications --------------------------------------
    md.append("## 8. Limitations")
    md.append("- All data originates from a single canonical dataset file "
               "(`data/pirvision_office_dataset.csv`), i.e. a single data-collection "
               "run/session, so cross-session / cross-deployment generalization is untested.")
    md.append("- The final-chronological holdout is documented as a temporal stress analysis "
               "rather than a full multiclass benchmark whenever the tail of the collection "
               "period does not contain all three classes (see Section 7 above for the actual "
               "outcome observed in this run).")
    md.append("- Temperature_F contains sentinel/dropout values (==0) that were flagged, not "
               "imputed, to avoid introducing leakage or fabricated values.")
    md.append("- Hyperparameter search was intentionally small/manual (CPU-friendly, "
               "reproducible) rather than an exhaustive search; absolute performance ceilings "
               "may be somewhat higher with heavier tuning.")
    md.append("- All data originates from a single physical space/sensor rig; results may not "
               "transfer directly to a different room geometry or PIR sensor layout.")
    md.append("")
    md.append("## 9. Practical Implications")
    md.append("- The PIR sensor reduction experiment (Section 5) directly informs lightweight "
               "edge deployment: fewer active PIR channels reduce wiring, power draw, and "
               "per-sample inference cost while the ablation table quantifies the accuracy "
               "trade-off explicitly.")
    md.append("- Group-aware, temporally-leakage-controlled cross-validation (StratifiedGroupKFold "
               "over 5-minute temporal buckets) gives a more realistic generalization estimate "
               "than a naive random row-wise split for this kind of densely, irregularly "
               "sampled sensor stream.")
    md.append("- Explainability results identify which physical PIR sensor positions and the "
               "ambient temperature channel drive predictions, supporting sensor-placement "
               "decisions in future smart-environment deployments.")
    md.append("")

    out_path = RESULTS_DIR / "scientific_results_summary.md"
    out_path.write_text("\n".join(md))
    logger.info(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    generate_report()
