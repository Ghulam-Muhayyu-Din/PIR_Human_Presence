from __future__ import annotations

from PIL import Image

from src.config import FIGURES_DIR, RESULTS_DIR, FIG_DPI, get_logger
import pandas as pd

logger = get_logger("figure_qc")

REQUIRED_FIGURES = [
    "figure_01_workflow",
    "figure_02_class_distribution",
    "figure_03_model_comparison",
    "figure_04_confusion_matrix",
    "figure_05_shap_feature_importance",
    "figure_06_pir_sensor_reduction",
    "figure_07_performance_vs_cost",
    "figure_08_correlation_heatmap",
    "figure_09_per_class_performance",
    "figure_10_temporal_class_distribution",
]


def run_figure_qc() -> pd.DataFrame:
    rows = []

    for fig_name in REQUIRED_FIGURES:
        png_path = FIGURES_DIR / f"{fig_name}.png"
        pdf_path = FIGURES_DIR / f"{fig_name}.pdf"
        svg_path = FIGURES_DIR / f"{fig_name}.svg"

        if not png_path.exists():
            rows.append({
                "figure_name": fig_name,
                "file_path": str(png_path),
                "status": "MISSING",
                "width": None,
                "height": None,
                "dpi": None,
                "notes": "PNG file does not exist.",
            })
            continue

        if png_path.stat().st_size == 0:
            rows.append({
                "figure_name": fig_name,
                "file_path": str(png_path),
                "status": "EMPTY",
                "width": None,
                "height": None,
                "dpi": None,
                "notes": "PNG file is empty.",
            })
            continue

        try:
            with Image.open(png_path) as im:
                width, height = im.size
                dpi = im.info.get("dpi", (None, None))
                dpi_x = dpi[0] if dpi and dpi[0] else None
        except Exception as exc:
            rows.append({
                "figure_name": fig_name,
                "file_path": str(png_path),
                "status": "UNREADABLE",
                "width": None,
                "height": None,
                "dpi": None,
                "notes": f"PIL could not read image: {exc}",
            })
            continue

        notes = []
        status = "OK"

        if width < 300 or height < 300:
            status = "SUSPICIOUS_SMALL"
            notes.append(
                f"Small pixel dimensions: {width} by {height}."
            )

        if dpi_x and abs(dpi_x - FIG_DPI) > 1:
            notes.append(
                f"Reported DPI {dpi_x} differs from configured {FIG_DPI}."
            )

        if not pdf_path.exists():
            status = "INCOMPLETE" if status == "OK" else status
            notes.append("PDF companion missing.")

        if not svg_path.exists():
            status = "INCOMPLETE" if status == "OK" else status
            notes.append("SVG companion missing.")

        rows.append({
            "figure_name": fig_name,
            "file_path": str(png_path),
            "status": status,
            "width": width,
            "height": height,
            "dpi": dpi_x,
            "notes": "; ".join(notes) if notes else "OK",
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(
        RESULTS_DIR / "figure_audit.csv",
        index=False,
    )

    return audit_df


if __name__ == "__main__":
    run_figure_qc()
