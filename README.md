# Lightweight and Explainable Machine Learning for PIR-Based Human Presence Detection in Smart Environments

A complete, reproducible, research-grade ML pipeline for 3-class human-presence
classification (Vacancy / Stationary presence / Other activity) from a
55-channel PIR sensor array + ambient temperature, with temporal-leakage-aware
evaluation, empirical model selection, SHAP explainability, PIR sensor
reduction (ablation), and a computational-efficiency / accuracy trade-off
analysis for lightweight edge deployment.

## Dataset (single source of truth)

This project uses exactly **one** dataset file:
`data/pirvision_office_dataset.csv` (7651 rows: `Date`, `Time`, `Label`,
`Temperature_F`, `PIR_1`...`PIR_55`). There is no second/raw/duplicate
dataset anywhere in the pipeline -- `run_all.py` Step 0 only *validates*
that this single file exists, is readable, and is non-empty; no step ever
copies, downloads, or compares it against another file. `configs/config.yaml`
holds the single `data.canonical_csv` path that every module resolves
through `src/config.CANONICAL_CSV`.

## Project structure

```
PIR_Human_Presence/
  data/                        the single canonical dataset (pirvision_office_dataset.csv)
  src/
    config.py                  paths, seed, label mapping, logger factory
    utils.py                   PIR-column detection, timestamp/group helpers
    data_inspection.py         dataset audit -> results/dataset_audit.csv, dataset_summary.txt
    feature_identification.py  auto column-role detection -> results/feature_list.csv
    preprocessing.py           feature matrix build, scaling-pipeline factory
    model_training.py          model zoo + small hyperparameter grids
    cross_validation.py        principal 5-fold StratifiedGroupKFold CV
    evaluation.py               OOF report, final refit+save, efficiency profiling
    explainability.py          SHAP TreeExplainer / permutation-importance fallback
    sensor_ablation.py         top-K PIR sensor reduction experiment
    temporal_analysis.py       chronological analysis -> results/temporal_analysis.csv
    figures.py                 all publication figures (600 DPI, PNG+PDF+SVG)
    figure_qc.py                post-generation figure quality control -> results/figure_audit.csv
    report_generation.py       programmatic scientific_results_summary.md
  results/                     all CSV/TXT/JSON/MD outputs
  figures/                     all figure files
  models/                      best_model.joblib + best_model_config.json
  configs/config.yaml          seed, CV settings, paths, ablation sizes
  logs/pipeline.log            full run log
  run_all.py                   single entry point
  requirements.txt
  README.md
```

## How to run in PyCharm

1. Open the `PIR_Human_Presence` folder as a PyCharm project.
2. Create a virtual environment (PyCharm: *File > Settings > Project >
   Python Interpreter > Add Interpreter > Add Local Interpreter > venv*,
   base interpreter = Python 3.11), or from a terminal:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (In a locked-down/managed environment, e.g. this sandbox, use
   `pip install --break-system-packages -r requirements.txt` instead.)
4. Run the full pipeline (right-click `run_all.py` > Run, or from a terminal
   in the project root):
   ```bash
   python run_all.py
   ```
5. Outputs land in `results/`, `figures/`, `models/`, and `logs/pipeline.log`.
   Runtime is a few minutes on a modern CPU (whole pipeline, ~9-10 models x
   5 folds x small hyperparameter grids, plus ablation and efficiency
   profiling).

## Reproducibility

- A single seed (`42`) is set globally (`src/config.set_global_seed`) and
  passed to every estimator, CV splitter, and sampling operation.
- The principal evaluation strategy is **StratifiedGroupKFold** (5 folds),
  grouped by 5-minute temporal buckets per date (see
  `src/utils.build_temporal_groups`), so temporally adjacent, highly
  correlated PIR readings never leak across train/validation.
- All preprocessing (StandardScaler for distance/gradient-sensitive models)
  is fit exclusively inside `sklearn.Pipeline` on the training fold.
- Every artifact required for reproducing or auditing a run is saved:
  `results/dataset_audit.csv`, `results/dataset_summary.txt`,
  `results/feature_list.csv`, `results/final_model_parameters.json`,
  `results/model_cv_summary.csv`, `results/model_fold_results.csv`,
  `results/oof_predictions_best.csv`, `results/classification_report_best.csv`,
  `results/feature_importance.csv`, `results/ablation_results.csv`,
  `results/computational_efficiency.csv`, `results/temporal_analysis.csv`,
  `results/figure_audit.csv`, `results/scientific_results_summary.md`,
  `models/best_model.joblib`, `models/best_model_config.json`, and
  `requirements.txt` pinning the tested library versions.

## Label convention (must-preserve)

The target column `Label` uses the **original values 0, 1, 3** (not 0,1,2):

| Label | Meaning                       |
|------:|--------------------------------|
| 0     | Vacancy                        |
| 1     | Stationary human presence      |
| 3     | Other activity / motion        |

Internally, scikit-learn/XGBoost/CatBoost require contiguous `0..k-1` class
indices, so the pipeline maps `0->0, 1->1, 3->2` only for model fitting
(`src/config.ORIG_TO_ENC` / `ENC_TO_ORIG`). Every exported result file
(reports, OOF predictions, confusion matrix, classification report) reports
back the **original** label values and their meanings.

## Graceful degradation

- If `catboost` fails to install/import, it is dropped from the model zoo
  with a logged warning; the pipeline still runs and selects the best model
  among the remaining candidates.
- If `shap` is unavailable (or `TreeExplainer` fails on the selected model),
  explainability automatically falls back to `sklearn.inspection.
  permutation_importance`, and this fallback is explicitly recorded in
  `results/feature_importance.csv` (`method` column) and in the console log.

## What each figure shows

| Figure | Filename | Content |
|---|---|---|
| 1 | `figure_01_workflow` | Research workflow schematic (11 pipeline stages) |
| 2 | `figure_02_class_distribution` | Class distribution bar chart with counts/percentages |
| 3 | `figure_03_model_comparison` | Macro F1 (+ Accuracy, Balanced Accuracy, MCC) per candidate model, sorted by performance |
| 4 | `figure_04_confusion_matrix` | Best-model normalized confusion matrix (OOF) |
| 5 | `figure_05_shap_feature_importance` | SHAP TreeExplainer (or permutation-importance fallback) feature importance |
| 6 | `figure_06_pir_sensor_reduction` | PIR sensor ablation: Macro F1 vs. #sensors, with error bars; full 55-sensor config marked |
| 7 | `figure_07_performance_vs_cost` | Macro F1 vs. training time trade-off, marker size = model size |
| 8 | `figure_08_correlation_heatmap` | Hierarchically-clustered PIR sensor correlation heatmap |
| 9 | `figure_09_per_class_performance` | Precision/Recall/F1 per class for the best model |
| 10 | `figure_10_temporal_performance` | Honest temporal class-distribution view (see note below) |

Every figure is exported as `.png` (600 DPI), `.pdf`, and `.svg`, and is
programmatically quality-checked after generation (`src/figure_qc.py` ->
`results/figure_audit.csv`: file existence, size, pixel dimensions, DPI).

**Note on figure_10**: the dataset spans only 3 distinct collection dates,
and the final chronological date (`2024-10-08`) contains a single class
(Vacancy only). A naive "OOF accuracy per date" line chart across 3 points
would be statistically thin and would trivially read as near-perfect on the
single-class day, which is misleading. `figure_10_temporal_performance`
instead shows the honest picture: class counts per date (stacked bars),
with the single-class chronological-holdout date explicitly flagged on the
chart. This mirrors the "temporal stress analysis, not a full benchmark"
finding documented in `results/temporal_analysis.csv`.

## Research integrity note

No results, citations, or novelty claims in this project are fabricated.
`results/scientific_results_summary.md` is built programmatically by reading
back the pipeline's own generated CSV/JSON files — it is not hand-typed.
