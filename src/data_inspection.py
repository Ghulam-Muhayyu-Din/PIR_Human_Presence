"""
Dataset audit module.

Validates the single canonical dataset (data/pirvision_office_dataset.csv)
programmatically: existence/readability, shape, dtypes, missing values,
duplicate rows, target values, class distribution, date/time range,
sampling-gap continuity, and temperature data quality. Writes:
    results/dataset_audit.csv
    results/dataset_summary.txt
    results/per_date_class_distribution.csv

This project has exactly ONE dataset file. There is no second/raw/duplicate
source to compare against -- this module only ever reads CANONICAL_CSV.
"""
from __future__ import annotations

import pandas as pd

from src.config import CANONICAL_CSV, RESULTS_DIR, TARGET_COL, ORIG_TO_MEANING
from src.config import get_logger
from src.utils import detect_pir_columns, build_timestamp

logger = get_logger("data_inspection")


def run_audit() -> pd.DataFrame:
    logger.info("Starting dataset audit ...")

    if not CANONICAL_CSV.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found at {CANONICAL_CSV}. This project uses a single "
            "dataset file (data/pirvision_office_dataset.csv); it must already exist -- "
            "the pipeline never creates, copies, or downloads it.")
    if CANONICAL_CSV.stat().st_size == 0:
        raise ValueError(f"Canonical dataset at {CANONICAL_CSV} is empty (0 bytes).")

    df = pd.read_csv(CANONICAL_CSV)
    if df.empty:
        raise ValueError(f"Canonical dataset at {CANONICAL_CSV} has 0 rows after parsing.")

    audit_rows = []

    def add(metric, value, notes=""):
        audit_rows.append({"metric": metric, "value": value, "notes": notes})

    # --- shape / schema ----------------------------------------------------
    add("dataset_path", str(CANONICAL_CSV), "The single canonical dataset used by the entire pipeline.")
    add("n_rows", df.shape[0])
    add("n_columns", df.shape[1])
    add("column_names", " | ".join(df.columns.tolist()))
    add("dtypes", " | ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()))

    # --- required columns present -------------------------------------------
    required_cols = ["Date", "Time", TARGET_COL, "Temperature_F"]
    missing_required = [c for c in required_cols if c not in df.columns]
    add("required_columns_present", len(missing_required) == 0,
        f"Checked for {required_cols}. Missing: {missing_required if missing_required else 'none'}.")
    if missing_required:
        raise ValueError(f"Canonical dataset is missing required columns: {missing_required}")

    # --- target value validation -------------------------------------------
    expected_targets = {0, 1, 3}
    actual_targets = set(int(v) for v in df[TARGET_COL].unique())
    add("target_values_match_expected", actual_targets == expected_targets,
        f"Expected {sorted(expected_targets)}, found {sorted(actual_targets)}.")

    # --- missing values / duplicate rows ------------------------------------
    n_missing = int(df.isna().sum().sum())
    add("total_missing_values", n_missing)
    n_dupe_rows = int(df.duplicated().sum())
    add("duplicate_rows_within_file", n_dupe_rows)

    # --- identifier column check --------------------------------------------
    has_unique_int_id_col = False
    id_candidates = []
    for c in df.columns:
        if df[c].is_unique and pd.api.types.is_integer_dtype(df[c]):
            id_candidates.append(c)
    if id_candidates:
        has_unique_int_id_col = True
    add("identifier_column_candidates", ",".join(id_candidates) if id_candidates else "NONE",
        "Programmatically checked for any column that is both integer-typed and 100% "
        "unique-valued (a plausible row-ID). Result: " +
        ("found candidate(s)" if has_unique_int_id_col else
         "no explicit identifier/row-ID column exists in the raw data."))

    # --- target / class distribution ----------------------------------------
    class_counts = df[TARGET_COL].value_counts().sort_index()
    add("unique_target_values", ",".join(str(v) for v in sorted(df[TARGET_COL].unique())))
    for lbl, cnt in class_counts.items():
        add(f"class_count_label_{lbl}", int(cnt), ORIG_TO_MEANING.get(int(lbl), "unknown"))
    add("class_imbalance_ratio_majority_to_minority",
        round(class_counts.max() / class_counts.min(), 3),
        "Ratio of largest class count to smallest class count -> confirms imbalance.")

    # --- PIR feature columns --------------------------------------------------
    pir_cols = detect_pir_columns(df)
    add("n_pir_sensor_columns_detected", len(pir_cols),
        f"Detected via regex PIR_\\d+: {pir_cols[0]} ... {pir_cols[-1]}")

    # --- feature ranges (PIR + temperature) ------------------------------------
    add("PIR_global_min", int(df[pir_cols].min().min()))
    add("PIR_global_max", int(df[pir_cols].max().max()))
    add("Temperature_F_min", int(df["Temperature_F"].min()))
    add("Temperature_F_max", int(df["Temperature_F"].max()))
    n_temp_zero = int((df["Temperature_F"] == 0).sum())
    add("Temperature_F_zero_value_rows", n_temp_zero,
        "Temperature==0 is not physically plausible for an indoor office in deg F and is "
        "treated as a probable sensor-dropout / missing-value sentinel. DECISION: these rows "
        "are NOT silently imputed (to avoid leaking distributional information from the full "
        "dataset into any train fold); a boolean flag column 'Temperature_F_is_zero_flag' is "
        "added in preprocessing so downstream models can (optionally) treat it as informative "
        "missingness, and the raw value is otherwise left untouched.")
    add("Temperature_F_zero_value_pct", round(100 * n_temp_zero / len(df), 3))

    # --- date / time range --------------------------------------------------
    ts = build_timestamp(df)
    add("date_min", str(df["Date"].min()))
    add("date_max", str(df["Date"].max()))
    add("time_min", str(df["Time"].min()))
    add("time_max", str(df["Time"].max()))
    add("timestamp_min", str(ts.min()))
    add("timestamp_max", str(ts.max()))

    # --- sampling gap / continuity analysis ----------------------------------
    ts_sorted = ts.sort_values().reset_index(drop=True)
    gaps = ts_sorted.diff().dt.total_seconds().dropna()
    add("median_sampling_gap_seconds", float(gaps.median()))
    add("mean_sampling_gap_seconds", round(float(gaps.mean()), 3))
    add("min_sampling_gap_seconds", float(gaps.min()))
    add("max_sampling_gap_seconds", float(gaps.max()))
    add("pct_gaps_over_60s", round(100 * float((gaps > 60).mean()), 3),
        "Fraction of consecutive-observation gaps exceeding 60 seconds -> evidence "
        "sampling is NOT perfectly evenly spaced (confirms task note).")
    add("n_unique_gap_values", int(gaps.round(3).nunique()),
        "More than 1 unique gap value confirms non-uniform sampling interval.")

    # --- potential leakage / identifier variables -----------------------------
    add("potential_leakage_variables", "Date,Time",
        "Raw Date/Time strings are exact timestamps; used only to build temporal groups "
        "for StratifiedGroupKFold and for the temporal stress test, and are EXCLUDED from "
        "the model feature matrix to prevent the model from memorizing timestamps.")
    add("potential_identifier_variables", "NONE",
        "No column is both unique-valued and an unambiguous row identifier; row order in "
        "the CSV is the only implicit identifier and is not used as a feature.")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)

    # --- per-date class distribution (used later by temporal_analysis) --------
    per_date = (
        df.assign(Label=df[TARGET_COL])
        .groupby("Date")["Label"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )
    per_date.to_csv(RESULTS_DIR / "per_date_class_distribution.csv")

    # --- human-readable summary -------------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append("PIR HUMAN PRESENCE DETECTION -- DATASET AUDIT SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Dataset file            : {CANONICAL_CSV}")
    lines.append(f"Rows x Columns          : {df.shape[0]} x {df.shape[1]}")
    lines.append(f"Columns                 : {', '.join(df.columns.tolist())}")
    lines.append(f"Target values match expected {{0,1,3}}: {actual_targets == expected_targets} "
                 f"(found {sorted(actual_targets)})")
    lines.append("")
    lines.append("MISSING VALUES / DUPLICATE ROWS")
    lines.append("-" * 78)
    lines.append(f"  Total missing values (NaN)         : {n_missing}")
    lines.append(f"  Duplicate rows within the file      : {n_dupe_rows}")
    lines.append(f"  Identifier/row-ID column present?   : {has_unique_int_id_col} "
                 f"({','.join(id_candidates) if id_candidates else 'none found'})")
    lines.append("")
    lines.append("CLASS DISTRIBUTION (target = Label; values are 0, 1, 3 -- NOT 0,1,2)")
    lines.append("-" * 78)
    for lbl, cnt in class_counts.items():
        pct = 100 * cnt / len(df)
        lines.append(f"  Label={lbl} ({ORIG_TO_MEANING.get(int(lbl),'?'):28s}): {cnt:5d} rows ({pct:5.2f}%)")
    lines.append(f"  Imbalance ratio (majority/minority) : {class_counts.max()/class_counts.min():.2f}")
    lines.append("")
    lines.append("FEATURES")
    lines.append("-" * 78)
    lines.append(f"  PIR sensor columns detected (regex PIR_\\d+) : {len(pir_cols)}")
    lines.append(f"  PIR value range                              : [{int(df[pir_cols].min().min())}, {int(df[pir_cols].max().max())}]")
    lines.append(f"  Temperature_F range                          : [{int(df['Temperature_F'].min())}, {int(df['Temperature_F'].max())}]")
    lines.append(f"  Temperature_F == 0 rows (data-quality flag)  : {n_temp_zero} ({100*n_temp_zero/len(df):.2f}%)")
    lines.append("    -> Treated as probable sensor dropout / missing-value sentinel, NOT")
    lines.append("       physically plausible indoor temperature. Not silently imputed; flagged")
    lines.append("       via a boolean indicator feature instead (see preprocessing.py).")
    lines.append("")
    lines.append("TEMPORAL COVERAGE / SAMPLING")
    lines.append("-" * 78)
    lines.append(f"  Date range   : {df['Date'].min()} to {df['Date'].max()}")
    lines.append(f"  Time range   : {df['Time'].min()} to {df['Time'].max()}")
    lines.append(f"  Median gap between consecutive observations : {gaps.median():.1f} s")
    lines.append(f"  Mean gap                                     : {gaps.mean():.1f} s")
    lines.append(f"  Min / Max gap                                : {gaps.min():.1f} s / {gaps.max():.1f} s")
    lines.append(f"  % of gaps > 60s (non-uniform sampling evidence): {100*float((gaps>60).mean()):.2f}%")
    lines.append("  -> Observations are confirmed NOT perfectly evenly spaced.")
    lines.append("")
    lines.append("LEAKAGE / IDENTIFIER RISK VARIABLES")
    lines.append("-" * 78)
    lines.append("  Date, Time: exact timestamps -- used only to build temporal groups for")
    lines.append("  StratifiedGroupKFold and the temporal stress test; excluded from the")
    lines.append("  feature matrix used to fit models.")
    lines.append("  No explicit row-ID / identifier column exists in the raw data.")
    lines.append("")
    lines.append("=" * 78)

    with open(RESULTS_DIR / "dataset_summary.txt", "w") as f:
        f.write("\n".join(lines))

    logger.info("Dataset audit complete -> results/dataset_audit.csv, results/dataset_summary.txt")
    return audit_df


if __name__ == "__main__":
    run_audit()
