"""
Publication-quality figure generation module.

All figures: 600 DPI PNG + PDF + SVG, colorblind-safe seaborn palette, one
consistent typography system, constrained_layout (falling back to a tuned
tight_layout where a figure needs manual twin-axis handling), no 3D/pie
charts, minimal gridlines.

Typography (applied globally via rcParams below):
    title          15-17pt bold
    axis labels    12-14pt
    tick labels    10-12pt
    legend         10-12pt
    annotations    9-11pt
Font family: DejaVu Sans (bundled with Matplotlib -- always available in the
sandbox; no external/system font is referenced so this never silently
degrades).

Figure numbering (exact filenames, see also explainability.py for figure_05):
    figure_01_workflow                  research workflow schematic
    figure_02_class_distribution        class distribution bar chart
    figure_03_model_comparison          model comparison (Macro F1 + other metrics)
    figure_04_confusion_matrix          best-model normalized confusion matrix (OOF)
    figure_05_shap_feature_importance   SHAP / permutation-importance summary
                                         (produced in explainability.py)
    figure_06_pir_sensor_reduction      PIR sensor reduction (macro F1 vs #sensors)
    figure_07_performance_vs_cost       performance vs computational cost
    figure_08_correlation_heatmap       PIR sensor correlation heatmap
    figure_09_per_class_performance     per-class precision/recall/F1
    figure_10_temporal_class_distribution honest temporal class-distribution view
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import FIGURES_DIR, FIG_DPI, FIG_FORMATS, PALETTE, ORIG_TO_MEANING, get_logger

logger = get_logger("figures")

# ---------------------------------------------------------------------------
# One consistent typography / style system for every figure in the project.
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COLORS = sns.color_palette(PALETTE)


def _assert_pir_only(X: pd.DataFrame, pir_cols: list[str] | None = None):
    """Guard primary figures against accidental inclusion of non PIR features."""
    cols = list(X.columns)
    non_pir = [c for c in cols if not str(c).startswith("PIR_")]
    if non_pir:
        raise ValueError(
            "Primary PIR-only figures received non PIR features: "
            f"{non_pir}"
        )
    if pir_cols is not None and set(cols) != set(pir_cols):
        raise ValueError(
            "X columns and pir_cols differ in the PIR-only figure pipeline."
        )


def _save(fig, name: str, use_tight: bool = True):
    for fmt in FIG_FORMATS:
        kwargs = {"dpi": FIG_DPI}
        if use_tight:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", **kwargs)
    plt.close(fig)
    logger.info(f"Saved figure: {name} ({', '.join(FIG_FORMATS)})")


def _avoid_overlaps(xs, ys, min_sep_frac=0.045):
    """Very small manual 'adjustText'-style declutter: given data-space y
    values (already normalized 0-1 axis fraction) sorted by x, nudge labels
    that are within min_sep_frac of each other apart vertically. Returns a
    list of y-offsets (in axis fraction) to apply to each label."""
    order = np.argsort(ys)
    y_sorted = np.array(ys, dtype=float)[order]
    offsets = np.zeros(len(ys))
    adjusted = y_sorted.copy()
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_sep_frac:
            adjusted[i] = adjusted[i - 1] + min_sep_frac
    delta = adjusted - y_sorted
    offsets[order] = delta
    return offsets


# ============================================================================
def figure_01_workflow():
    """Fig1: full pipeline as boxes/arrows, wrapped across two rows so no box
    text is ever cramped and no arrow crosses another box."""
    steps_row1 = [
        "Dataset\n(single canonical\nCSV)",
        "Data audit",
        "Preprocessing",
        "Temporal\ngrouping",
        "Cross\nvalidation",
        "Model\ncomparison",
    ]
    steps_row2 = [
        "Best model",
        "SHAP\nexplainability",
        "Sensor\nreduction",
        "Computational\nanalysis",
        "Final results",
    ]

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box_w, box_h = 1.72, 1.55

    def _draw_row(steps, y0):
        n = len(steps)
        gap = (12 - n * box_w) / (n + 1)
        xs = []
        for i, label in enumerate(steps):
            x = gap + i * (box_w + gap)
            xs.append(x)
            color = COLORS[i % len(COLORS)]
            box = FancyBboxPatch((x, y0), box_w, box_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.20)
            ax.add_patch(box)
            ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
                     fontsize=10.5, fontweight="bold", color="#1a1a1a", linespacing=1.35)
        for i in range(n - 1):
            arrow = FancyArrowPatch((xs[i] + box_w, y0 + box_h / 2),
                                     (xs[i + 1], y0 + box_h / 2),
                                     arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
                                     color="#444444")
            ax.add_patch(arrow)
        return xs

    y_row1 = 4.15
    y_row2 = 1.15
    xs1 = _draw_row(steps_row1, y_row1)
    xs2 = _draw_row(steps_row2, y_row2)

    # connector: end of row1 down to start of row2
    x_end = xs1[-1] + box_w
    x_start2 = xs2[0]
    connector = FancyArrowPatch(
        (x_end - box_w / 2, y_row1), (x_start2 + box_w / 2, y_row2 + box_h),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#444444",
        connectionstyle="arc3,rad=-0.25")
    ax.add_patch(connector)

    ax.set_title("Research Workflow: PIR-Based Human Presence Detection",
                  fontsize=16, fontweight="bold", pad=18)
    fig.subplots_adjust(top=0.88)
    _save(fig, "figure_01_workflow", use_tight=False)


# ============================================================================
def figure_02_class_distribution(y_original: np.ndarray):
    """Fig2: class counts with headroom for value labels and non-colliding,
    non-rotated tick labels (two-line, wrapped)."""
    vc = pd.Series(y_original).value_counts().sort_index()
    meanings = {0: "Vacancy", 1: "Stationary\npresence", 3: "Other activity\n/ motion"}
    labels = [f"Label {v}\n{meanings.get(v, ORIG_TO_MEANING.get(v, '?'))}" for v in vc.index]

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    bars = ax.bar(labels, vc.values, color=COLORS[:len(vc)], edgecolor="black", linewidth=0.7,
                   width=0.6)
    ymax = max(vc.values)
    ax.set_ylim(0, ymax * 1.20)
    for b, v in zip(bars, vc.values):
        pct = 100 * v / vc.sum()
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.025,
                f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10.5)
    ax.set_xlabel("Target class")
    ax.set_ylabel("Number of observations")
    ax.set_title("Class Distribution — Imbalanced 3-Class Target", pad=14)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "figure_02_class_distribution")


# ============================================================================
def figure_03_model_comparison(cv_summary_df: pd.DataFrame):
    """Focused model comparison using the primary selection metric and MCC."""
    df = cv_summary_df.sort_values("macro_f1_mean", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    y = np.arange(len(df))

    ax.errorbar(
        df["macro_f1_mean"],
        y,
        xerr=df["macro_f1_std"] if "macro_f1_std" in df.columns else None,
        fmt="o",
        markersize=7,
        linewidth=1.8,
        capsize=4,
        color=COLORS[0],
        label="Macro F1 mean plus or minus standard deviation",
        zorder=4,
    )

    ax.scatter(
        df["mcc_mean"],
        y,
        marker="D",
        s=46,
        color=COLORS[2],
        edgecolor="black",
        linewidth=0.6,
        label="MCC mean",
        zorder=5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Score")
    ax.set_xlim(0.80, 1.01)
    ax.set_title(
        "Model Comparison: Macro F1 and MCC",
        pad=16,
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.22)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10,
        borderaxespad=0.6,
    )

    best_idx = df["macro_f1_mean"].idxmax()
    best_pos = df.index.get_loc(best_idx)
    best_row = df.loc[best_idx]
    ax.annotate(
        f"Best Macro F1: {best_row['macro_f1_mean']:.4f}",
        xy=(best_row["macro_f1_mean"], best_pos),
        xytext=(-72, 22),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.0,
            color="#444444",
        ),
    )

    fig.tight_layout()
    _save(fig, "figure_03_model_comparison")

# ============================================================================
def figure_04_confusion_matrix(y_true_original: np.ndarray, y_pred_original: np.ndarray,
                                best_model_name: str):
    """Fig4: normalized confusion matrix, colorbar with its own label, title
    separated from the axes with padding."""
    labels = sorted(pd.unique(np.concatenate([y_true_original, y_pred_original])).tolist())
    cm = confusion_matrix(y_true_original, y_pred_original, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    tick_labels = [f"{l}: {ORIG_TO_MEANING.get(l, '?')}" for l in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                cbar_kws={"label": "Row-normalized proportion", "shrink": 0.85},
                xticklabels=tick_labels, yticklabels=tick_labels, ax=ax,
                vmin=0, vmax=1, linewidths=0.6, linecolor="white",
                annot_kws={"fontsize": 12})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (OOF) — {best_model_name}", pad=14, fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save(fig, "figure_04_confusion_matrix")


# ============================================================================
def figure_06_pir_sensor_reduction(ablation_df: pd.DataFrame):
    """PIR sensor reduction using Macro F1 as the principal performance measure."""
    df = ablation_df.sort_values("n_pir_sensors").copy()

    full_row = df.iloc[-1]
    thirty_row = df.loc[df["n_pir_sensors"] == 30].iloc[0]
    retention = 100 * thirty_row["macro_f1_mean"] / full_row["macro_f1_mean"]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    ax.plot(
        df["n_pir_sensors"],
        df["macro_f1_mean"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=COLORS[0],
        label="Macro F1 mean",
        zorder=3,
    )

    ax.fill_between(
        df["n_pir_sensors"].to_numpy(),
        (df["macro_f1_mean"] - df["macro_f1_std"]).to_numpy(),
        (df["macro_f1_mean"] + df["macro_f1_std"]).to_numpy(),
        color=COLORS[0],
        alpha=0.14,
        label="Fold variation",
        zorder=1,
    )

    ax.scatter(
        [full_row["n_pir_sensors"]],
        [full_row["macro_f1_mean"]],
        marker="*",
        s=260,
        color="#C23B22",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Full configuration",
    )

    ax.scatter(
        [thirty_row["n_pir_sensors"]],
        [thirty_row["macro_f1_mean"]],
        marker="s",
        s=70,
        color=COLORS[2],
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="30 sensor configuration",
    )

    ax.annotate(
        f"30 sensors retain {retention:.1f}% of full Macro F1",
        xy=(
            thirty_row["n_pir_sensors"],
            thirty_row["macro_f1_mean"],
        ),
        xytext=(16, -38),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.0,
            color="#444444",
        ),
    )

    ax.set_xlabel("Number of retained PIR sensors")
    ax.set_ylabel("Macro F1, five fold group aware CV")
    ax.set_title(
        "PIR Sensor Reduction and Classification Performance",
        pad=16,
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(df["n_pir_sensors"])
    ax.set_ylim(
        max(0.84, (df["macro_f1_mean"] - df["macro_f1_std"]).min() - 0.01),
        min(0.99, (df["macro_f1_mean"] + df["macro_f1_std"]).max() + 0.01),
    )
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9.5,
    )

    fig.tight_layout()
    _save(fig, "figure_06_pir_sensor_reduction")

# ============================================================================
def figure_07_performance_vs_cost(eff_df: pd.DataFrame):
    """Model performance versus training cost with an external model legend."""
    df = eff_df.copy()

    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    x = df["train_time_sec_full_data"].to_numpy()
    y = df["macro_f1_cv_mean"].to_numpy()

    sizes = (
        80
        + 120
        * (
            df["model_size_kb"]
            / df["model_size_kb"].max()
        ).clip(lower=0.05)
    )

    handles = []

    for i, (_, row) in enumerate(df.iterrows()):
        color = COLORS[i % len(COLORS)]

        scatter = ax.scatter(
            row["train_time_sec_full_data"],
            row["macro_f1_cv_mean"],
            s=float(sizes.iloc[i]),
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.88,
            zorder=3,
        )

        handles.append(
            matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=7,
                label=row["model"],
            )
        )

    best_row = df.loc[df["macro_f1_cv_mean"].idxmax()]

    ax.scatter(
        [best_row["train_time_sec_full_data"]],
        [best_row["macro_f1_cv_mean"]],
        marker="*",
        s=300,
        facecolor="none",
        edgecolor="#C23B22",
        linewidth=1.5,
        zorder=6,
    )

    ax.annotate(
        f"Best Macro F1: {best_row['macro_f1_cv_mean']:.4f}",
        xy=(
            best_row["train_time_sec_full_data"],
            best_row["macro_f1_cv_mean"],
        ),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.0,
            color="#444444",
        ),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Training time on full dataset, seconds, logarithmic scale")
    ax.set_ylabel("Macro F1, five fold CV mean")
    ax.set_title(
        "Model Performance and Computational Cost",
        pad=16,
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(alpha=0.22)

    ax.legend(
        handles=handles,
        title="Model",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9.5,
        title_fontsize=10.5,
    )

    fig.tight_layout()
    _save(fig, "figure_07_performance_vs_cost")

# ============================================================================
def figure_08_correlation_heatmap(X: pd.DataFrame, pir_cols: list):
    _assert_pir_only(X, pir_cols)
    """Fig8: PIR sensor correlation structure.

    DESIGN CHOICE (documented): with 55 PIR sensors, printing all 55 x 55
    tick labels is not legible at any reasonable figure size/DPI. We use a
    scientifically-justified alternative: (1) hierarchical clustering
    (average-linkage on 1-|corr| distance) reorders sensors so physically
    adjacent / highly-correlated sensors form visible contiguous blocks,
    and (2) tick labels are thinned to every 4th sensor (in cluster order)
    to stay legible while still anchoring the reader to sensor identity.
    """
    corr = X[pir_cols].corr()

    dist = 1 - corr.abs()
    condensed = dist.values[np.triu_indices_from(dist.values, k=1)]
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    ordered_cols = [pir_cols[i] for i in order]
    corr_ordered = corr.loc[ordered_cols, ordered_cols]

    n = len(ordered_cols)
    tick_every = 4
    tick_idx = list(range(0, n, tick_every))
    tick_labels = [ordered_cols[i] for i in tick_idx]

    fig, ax = plt.subplots(figsize=(9, 7.6))
    sns.heatmap(corr_ordered, cmap="coolwarm", center=0, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation", "shrink": 0.8}, linewidths=0)
    ax.set_xticks([i + 0.5 for i in tick_idx])
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8.5)
    ax.set_yticks([i + 0.5 for i in tick_idx])
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=8.5)
    ax.set_title("PIR Sensor Correlation Heatmap\n(hierarchically clustered; every 4th of 55 sensors labeled)",
                 pad=16, fontsize=14.5)
    fig.tight_layout()
    _save(fig, "figure_08_correlation_heatmap")


# ============================================================================
def figure_09_per_class_performance(cv_summary_df: pd.DataFrame, best_model_name: str):
    row = cv_summary_df[cv_summary_df["model"] == best_model_name].iloc[0]
    classes = []
    for c in row.index:
        if c.startswith("f1_class_") and c.endswith("_mean"):
            classes.append(c.replace("f1_class_", "").replace("_mean", ""))
    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(classes))
    bar_w = 0.25
    for i, m in enumerate(metrics):
        vals = [row[f"{m}_class_{c}_mean"] for c in classes]
        ax.bar(x + (i - 1) * bar_w, vals, width=bar_w, label=m.capitalize(),
               color=COLORS[i % len(COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{ORIG_TO_MEANING.get(int(c), '?')}" for c in classes])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Performance — {best_model_name} (CV mean)", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "figure_09_per_class_performance")


# ============================================================================
def figure_10_temporal_performance(df: pd.DataFrame, temporal_result: dict):
    """Temporal class distribution view used to document collection coverage."""
    per_date = (
        df.groupby("Date")["Label"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    per_date.columns = [
        int(c)
        for c in per_date.columns
    ]

    class_order = [
        c
        for c in (0, 1, 3)
        if c in per_date.columns
    ]

    color_map = {
        0: COLORS[0],
        1: COLORS[1],
        3: COLORS[2],
    }

    label_map = {
        0: "Vacancy",
        1: "Stationary presence",
        3: "Other activity and motion",
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.9))

    x = np.arange(len(per_date))
    bottom = np.zeros(len(per_date))

    for c in class_order:
        vals = per_date[c].to_numpy()

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=label_map[c],
            color=color_map[c],
            edgecolor="black",
            linewidth=0.5,
        )

        bottom += vals

    totals = per_date[class_order].sum(axis=1)

    ax.set_ylim(
        0,
        totals.max() * 1.30,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        per_date.index.tolist()
    )

    ax.set_xlabel("Collection date")
    ax.set_ylabel("Number of observations")

    ax.set_title(
        "Temporal Class Distribution by Collection Date",
        pad=16,
        fontsize=16,
        fontweight="bold",
    )

    holdout_counts = temporal_result.get(
        "holdout_row_counts_by_date",
        {},
    )

    for xi, date in zip(
        x,
        per_date.index.tolist(),
    ):

        n_holdout = holdout_counts.get(
            date,
            0,
        )

        if n_holdout == 0:
            continue

        total = int(
            totals.loc[date]
        )

        n_classes_present = int(
            (
                per_date.loc[
                    date,
                    class_order,
                ]
                > 0
            ).sum()
        )

        if n_holdout == total:

            note = (
                "Chronological holdout\n"
                f"{'single class' if n_classes_present == 1 else f'{n_classes_present} classes'}"
            )

        else:

            note = (
                "Chronological holdout\n"
                f"final {n_holdout:,} of {total:,} rows"
            )

        ax.annotate(
            note,
            (xi, total),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#9E2A2B",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#9E2A2B",
                linewidth=0.8,
                alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.9,
                color="#9E2A2B",
            ),
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        title="Class",
        fontsize=9.5,
        title_fontsize=10.5,
    )

    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    _save(
        fig,
        "figure_10_temporal_class_distribution",
    )

