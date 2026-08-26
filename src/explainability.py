"""
Explainability for the primary PIR only experiment.

The explainability output contains PIR sensor features only. Temperature is
never included in the primary feature importance analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    FIG_DPI,
    FIG_FORMATS,
    PALETTE,
    SEED,
    get_logger,
)

logger = get_logger("explainability")

TREE_MODEL_NAMES = {
    "ExtraTrees",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "HistGradientBoosting",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 11,
})


def _save_fig(fig, base_name: str):
    for fmt in FIG_FORMATS:
        path = FIGURES_DIR / f"{base_name}.{fmt}"
        fig.savefig(
            path,
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def _render_importance_figure(
    importance_df: pd.DataFrame,
    xlabel: str,
    title: str,
):
    top = (
        importance_df
        .head(20)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(
        figsize=(8.4, 6.8),
        constrained_layout=True,
    )

    ax.barh(
        top["feature"],
        top["importance"],
        color=sns.color_palette(PALETTE)[0],
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=12,
        labelpad=8,
    )
    ax.set_ylabel(
        "PIR sensor",
        fontsize=12,
    )
    ax.set_title(
        title,
        fontsize=15.5,
        fontweight="bold",
        pad=14,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )
    ax.grid(
        axis="y",
        visible=False,
    )

    _save_fig(
        fig,
        "figure_05_shap_feature_importance",
    )


def run_explainability(
    pipe,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    best_model_name: str,
    sample_n: int = 1500,
):
    # Hard safety check: primary explainability must be PIR only.
    non_pir = [
        c for c in X.columns
        if not str(c).startswith("PIR_")
    ]

    if non_pir:
        raise ValueError(
            "Non PIR features detected in primary explainability input: "
            f"{non_pir}"
        )

    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
    )

    method_used = None
    importance_df = None

    if best_model_name in TREE_MODEL_NAMES:
        try:
            import shap

            model = pipe.named_steps["model"]

            rng = np.random.RandomState(
                SEED
            )

            idx = rng.choice(
                len(X),
                size=min(
                    sample_n,
                    len(X),
                ),
                replace=False,
            )

            X_sample = X.iloc[idx]

            explainer = shap.TreeExplainer(
                model
            )

            shap_values = explainer.shap_values(
                X_sample
            )

            if isinstance(shap_values, list):

                class_importance = [
                    np.abs(sv).mean(axis=0)
                    for sv in shap_values
                ]

                mean_abs_shap = np.mean(
                    np.stack(
                        class_importance,
                        axis=0,
                    ),
                    axis=0,
                )

            else:

                sv_arr = np.asarray(
                    shap_values
                )

                if sv_arr.ndim == 3:
                    mean_abs_shap = (
                        np.abs(sv_arr)
                        .mean(axis=(0, 2))
                    )

                else:
                    mean_abs_shap = (
                        np.abs(sv_arr)
                        .mean(axis=0)
                    )

            importance_df = pd.DataFrame({
                "feature": X.columns,
                "importance": mean_abs_shap,
            }).sort_values(
                "importance",
                ascending=False,
            ).reset_index(
                drop=True
            )

            importance_df[
                "method"
            ] = "SHAP_TreeExplainer_mean_abs_shap"

            method_used = (
                "SHAP TreeExplainer"
            )

            _render_importance_figure(
                importance_df,
                xlabel=(
                    "Mean absolute SHAP value"
                ),
                title=(
                    f"PIR Sensor Importance: "
                    f"{best_model_name}"
                ),
            )

        except Exception as exc:

            logger.warning(
                "SHAP failed; falling back to permutation importance: "
                f"{exc}"
            )

    if method_used is None:

        from sklearn.inspection import (
            permutation_importance,
        )

        rng_idx = (
            np.random.RandomState(
                SEED
            ).choice(
                len(X),
                size=min(
                    sample_n,
                    len(X),
                ),
                replace=False,
            )
        )

        X_sample = X.iloc[
            rng_idx
        ]

        y_sample = y_encoded[
            rng_idx
        ]

        result = permutation_importance(
            pipe,
            X_sample,
            y_sample,
            n_repeats=10,
            random_state=SEED,
            scoring="f1_macro",
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values(
            "importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        importance_df[
            "method"
        ] = "permutation_importance_macro_f1_drop"

        method_used = (
            "Permutation importance"
        )

        _render_importance_figure(
            importance_df,
            xlabel=(
                "Mean decrease in Macro F1"
            ),
            title=(
                f"PIR Sensor Importance: "
                f"{best_model_name}"
            ),
        )

    # Final safety validation.
    if not importance_df["feature"].astype(str).str.startswith(
        "PIR_"
    ).all():
        raise RuntimeError(
            "Feature importance output contains a non PIR feature."
        )

    importance_df.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )

    top_pir = (
        importance_df
        .head(10)["feature"]
        .tolist()
    )

    logger.info(
        f"Explainability method: {method_used}"
    )

    logger.info(
        f"Top PIR sensors: {top_pir}"
    )

    return importance_df, method_used
