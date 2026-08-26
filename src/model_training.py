"""
Model zoo: defines every candidate model + a small, CPU-friendly
hyperparameter configuration for each, and builds a leakage-safe
sklearn Pipeline for each candidate.

Gracefully skips CatBoost if the package is unavailable (import guarded).
"""
from __future__ import annotations

from collections import OrderedDict

from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from src.config import SEED, get_logger
from src.preprocessing import make_pipeline

logger = get_logger("model_training")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:  # pragma: no cover
    HAS_XGBOOST = False
    logger.warning(f"xgboost unavailable, skipping: {e}")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception as e:  # pragma: no cover
    HAS_CATBOOST = False
    logger.warning(f"catboost unavailable, skipping: {e}")


def _class_weight_options():
    return [None, "balanced"]


def build_model_registry(n_jobs: int = -1) -> "OrderedDict[str, dict]":
    """
    Returns an ordered dict: model_name -> {
        "estimator_fn": callable(**params) -> unfitted estimator,
        "param_grid": list[dict] of small manual configs to try,
        "is_tree": bool,
    }
    Small manual grids only (brief section 7: keep search small/fast).
    """
    registry = OrderedDict()

    registry["ExtraTrees"] = {
        "estimator_fn": lambda **p: ExtraTreesClassifier(random_state=SEED, n_jobs=n_jobs, **p),
        "param_grid": [
            {"n_estimators": 300, "max_depth": None, "class_weight": cw}
            for cw in _class_weight_options()
        ],
        "is_tree": True,
    }

    registry["RandomForest"] = {
        "estimator_fn": lambda **p: RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **p),
        "param_grid": [
            {"n_estimators": 300, "max_depth": None, "class_weight": cw}
            for cw in _class_weight_options()
        ],
        "is_tree": True,
    }

    registry["DecisionTree"] = {
        "estimator_fn": lambda **p: DecisionTreeClassifier(random_state=SEED, **p),
        "param_grid": [
            {"max_depth": d, "class_weight": cw}
            for d in (None, 12)
            for cw in _class_weight_options()
        ],
        "is_tree": True,
    }

    registry["HistGradientBoosting"] = {
        "estimator_fn": lambda **p: HistGradientBoostingClassifier(random_state=SEED, **p),
        "param_grid": [
            {"max_iter": 200, "max_depth": None, "class_weight": cw}
            for cw in _class_weight_options()
        ],
        "is_tree": True,
    }

    if HAS_XGBOOST:
        registry["XGBoost"] = {
            "estimator_fn": lambda **p: XGBClassifier(
                random_state=SEED, n_jobs=n_jobs, eval_metric="mlogloss",
                verbosity=0, **p),
            "param_grid": [
                {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
                {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
            ],
            "is_tree": True,
        }

    if HAS_CATBOOST:
        registry["CatBoost"] = {
            "estimator_fn": lambda **p: CatBoostClassifier(
                random_state=SEED, verbose=False, **p),
            "param_grid": [
                {"iterations": 300, "depth": 6, "learning_rate": 0.1, "auto_class_weights": None},
                {"iterations": 300, "depth": 6, "learning_rate": 0.1, "auto_class_weights": "Balanced"},
            ],
            "is_tree": True,
        }

    registry["LogisticRegression"] = {
        "estimator_fn": lambda **p: LogisticRegression(
            random_state=SEED, max_iter=2000, **p),
        "param_grid": [
            {"C": C, "class_weight": cw}
            for C in (1.0, 0.1)
            for cw in _class_weight_options()
        ],
        "is_tree": False,
    }

    registry["SVM"] = {
        "estimator_fn": lambda **p: SVC(random_state=SEED, probability=True, **p),
        # RBF kernel, small param set -- kept small since SVM is the slowest model.
        "param_grid": [
            {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": cw}
            for cw in _class_weight_options()
        ],
        "is_tree": False,
    }

    registry["KNN"] = {
        "estimator_fn": lambda **p: KNeighborsClassifier(n_jobs=n_jobs, **p),
        "param_grid": [
            {"n_neighbors": k, "weights": "distance"}
            for k in (5, 15)
        ],
        "is_tree": False,
    }

    registry["MLP"] = {
        "estimator_fn": lambda **p: MLPClassifier(
            random_state=SEED, max_iter=300, early_stopping=True, **p),
        "param_grid": [
            {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
        ],
        "is_tree": False,
    }

    return registry


def build_pipeline_for(model_name: str, estimator):
    return make_pipeline(model_name, estimator)
