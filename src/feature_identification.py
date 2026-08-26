"""
Feature identification for the PIRvision primary experiment.

PIR sensors are the only primary model features.
Temperature is explicitly documented as excluded from the primary model.
Date and Time are used only for temporal grouping and never as model inputs.
"""
from __future__ import annotations

import pandas as pd

from src.config import CANONICAL_CSV, RESULTS_DIR, TARGET_COL, get_logger
from src.utils import detect_pir_columns

logger = get_logger("feature_identification")


def identify_features(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if df is None:
        df = pd.read_csv(CANONICAL_CSV)

    pir_cols = set(detect_pir_columns(df))
    rows = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        is_unique = df[col].is_unique
        n_unique = df[col].nunique()

        if col == TARGET_COL:
            role = "target"
            notes = (
                "Multiclass target; "
                f"unique values={sorted(df[col].unique().tolist())}"
            )

        elif col in pir_cols:
            role = "pir_sensor_feature"
            notes = (
                "Primary model feature; "
                f"range=[{df[col].min()},{df[col].max()}]"
            )

        elif col.lower().startswith("temperature"):
            role = "temperature_feature_excluded_from_primary_model"
            n_zero = int((df[col] == 0).sum())
            notes = (
                "Excluded from primary PIR only experiment. "
                f"Zero values={n_zero}. Retained in raw data only."
            )

        elif col.lower() in ("date", "time"):
            role = "datetime_component"
            notes = (
                "Used only to construct temporal ordering and CV groups; "
                "excluded from model input."
            )

        elif is_unique and pd.api.types.is_integer_dtype(df[col]):
            role = "identifier_candidate"
            notes = (
                "Integer typed and unique; not used as a model feature."
            )

        else:
            role = "other"
            notes = f"n_unique={n_unique}"

        rows.append({
            "feature_name": col,
            "role": role,
            "dtype": dtype,
            "n_unique": n_unique,
            "is_unique": is_unique,
            "notes": notes,
        })

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(
        RESULTS_DIR / "feature_list.csv",
        index=False,
    )

    n_pir = int(
        (feature_df["role"] == "pir_sensor_feature").sum()
    )

    logger.info(
        f"Feature identification complete: "
        f"{n_pir} PIR sensors; "
        f"target={TARGET_COL}; "
        "temperature explicitly excluded from primary model."
    )

    return feature_df


def get_model_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return only the PIR columns used by the primary experiment.
    """
    return detect_pir_columns(df)


if __name__ == "__main__":
    identify_features()
