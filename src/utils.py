"""Shared helper utilities used across the pipeline modules."""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import pandas as pd

from src.config import TARGET_COL

PIR_REGEX = re.compile(r"^PIR_\d+$")


def detect_pir_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if PIR_REGEX.match(c)]
    # sort numerically by sensor index rather than lexicographically
    cols = sorted(cols, key=lambda c: int(c.split("_")[1]))
    return cols


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                           format="%Y-%m-%d %H:%M:%S", errors="raise")


def seconds_since_midnight(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second


def build_temporal_groups(df: pd.DataFrame, window_seconds: int = 300) -> np.ndarray:
    """
    Build group ids for StratifiedGroupKFold: bucket each observation into a
    window_seconds-wide time bucket *within its date*, so that adjacent,
    highly time-correlated PIR readings are never split across train/val.
    Group id = date_string + "_" + bucket_index.
    """
    ts = build_timestamp(df)
    secs = seconds_since_midnight(ts)
    bucket = (secs // window_seconds).astype(int)
    group_key = df["Date"].astype(str) + "_" + bucket.astype(str)
    # factorize to integer group ids
    groups, _ = pd.factorize(group_key)
    return groups


@contextmanager
def timer():
    """Context manager yielding elapsed wall-clock seconds via .elapsed after exit."""
    class _T:
        elapsed = None
    t = _T()
    start = time.perf_counter()
    yield t
    t.elapsed = time.perf_counter() - start


def encode_labels(y_original: pd.Series, orig_to_enc: dict) -> np.ndarray:
    return y_original.map(orig_to_enc).to_numpy()


def decode_labels(y_encoded: np.ndarray, enc_to_orig: dict) -> np.ndarray:
    return np.array([enc_to_orig[v] for v in y_encoded])
