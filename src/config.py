"""
Central configuration and path management for the PIR Human Presence
Detection research pipeline.

All modules import PROJECT_ROOT and the various *_DIR constants from here
so that paths stay consistent regardless of which directory a script is
invoked from.
"""
from __future__ import annotations

import os
import random
import logging
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LOGS_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR, CONFIGS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

with open(CONFIGS_DIR / "config.yaml", "r") as _f:
    CONFIG = yaml.safe_load(_f)

SEED = int(CONFIG.get("seed", 42))
# The project has exactly ONE dataset file. Every module resolves this single
# path; there is no second/raw/duplicate dataset anywhere in the pipeline.
CANONICAL_CSV = PROJECT_ROOT / CONFIG["data"]["canonical_csv"]

TARGET_COL = CONFIG["target"]["column"]
ORIGINAL_LABELS = CONFIG["target"]["original_values"]          # [0, 1, 3]
ENCODED_LABELS = CONFIG["target"]["encoded_values"]             # [0, 1, 2]
LABEL_MAP = {int(k): v for k, v in CONFIG["target"]["label_map"].items()}
# original -> encoded
ORIG_TO_ENC = {int(k): v["encoded"] for k, v in CONFIG["target"]["label_map"].items()}
ENC_TO_ORIG = {v: k for k, v in ORIG_TO_ENC.items()}
ORIG_TO_MEANING = {int(k): v["meaning"] for k, v in CONFIG["target"]["label_map"].items()}

N_SPLITS = int(CONFIG["cv"]["n_splits"])
TEMPORAL_WINDOW_SECONDS = int(CONFIG["cv"]["temporal_group_window_seconds"])

PIR_SUBSET_SIZES = CONFIG["ablation"]["pir_subset_sizes"]

FIG_DPI = int(CONFIG["figures"]["dpi"])
FIG_FORMATS = CONFIG["figures"]["formats"]
PALETTE = CONFIG["figures"]["palette"]


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/pipeline.log and stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOGS_DIR / "pipeline.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger
