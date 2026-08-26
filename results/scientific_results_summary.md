# Scientific Results Summary

**Title:** Lightweight and Explainable Machine Learning for PIR Based Human Presence Detection in Smart Environments

_This document is generated programmatically from the pipeline's own result files (results/*.csv, results/*.json) -- no numbers below are hand-typed._

## 1. Dataset Description
- Single canonical dataset: `D:\Research\writing Article\Crosslink\PIR_Human_Presence\data\pirvision_office_dataset.csv`
- Rows x Columns: **7651 x 59**
- Date range: 2024-08-08 to 2024-10-08
- PIR sensor columns detected: **55**
- Target values match expected {0,1,3}: **True**
- Missing values: 0; duplicate rows within file: 0
- Identifier column present: NONE
- Temperature_F==0 rows (data-quality flag): 571 -- Temperature==0 is not physically plausible for an indoor office in deg F and is treated as a probable sensor-dropout / missing-value sentinel. DECISION: these rows are NOT silently imputed (to avoid leaking distributional information from the full dataset into any train fold); a boolean flag column 'Temperature_F_is_zero_flag' is added in preprocessing so downstream models can (optionally) treat it as informative missingness, and the raw value is otherwise left untouched.
- Median sampling gap between observations: 16.0 s (sampling confirmed non-uniform)

## 2. Class Distribution (target = Label; values 0, 1, 3)
- Label=0 (Vacancy): **6247** rows
- Label=1 (Stationary human presence): **833** rows
- Label=3 (Other activity/motion): **571** rows

## 3. Model Comparison (5-fold StratifiedGroupKFold CV, temporal-group-aware)
**Best model (empirically selected, primary metric = Macro F1): `RandomForest`**
- Macro F1  = 0.9197 (+/- 0.0234)
- Accuracy  = 0.9749
- Balanced Accuracy = 0.9122
- MCC       = 0.9199
- Weighted F1 = 0.9746

Full comparison table (best config per model, ranked by Macro F1):

| model                |   macro_f1_mean |   accuracy_mean |   balanced_accuracy_mean |   mcc_mean |   weighted_f1_mean |   mean_fit_time_sec |   mean_predict_time_sec |
|:---------------------|----------------:|----------------:|-------------------------:|-----------:|-------------------:|--------------------:|------------------------:|
| RandomForest         |          0.9197 |          0.9749 |                   0.9122 |     0.9199 |             0.9746 |              2.1811 |                  0.1815 |
| ExtraTrees           |          0.9189 |          0.9751 |                   0.9102 |     0.9211 |             0.9747 |              1.2087 |                  0.2102 |
| CatBoost             |          0.9162 |          0.9745 |                   0.9097 |     0.9191 |             0.9742 |              9.7287 |                  0.0095 |
| HistGradientBoosting |          0.9145 |          0.9734 |                   0.908  |     0.9158 |             0.9731 |              3.6506 |                  0.0746 |
| XGBoost              |          0.9118 |          0.9727 |                   0.903  |     0.9132 |             0.9722 |              4.3639 |                  0.0303 |
| SVM                  |          0.9087 |          0.9729 |                   0.8977 |     0.9142 |             0.9722 |              1.1152 |                  0.122  |
| MLP                  |          0.9038 |          0.9698 |                   0.89   |     0.9039 |             0.969  |              0.8806 |                  0.0037 |
| KNN                  |          0.8892 |          0.9667 |                   0.8781 |     0.895  |             0.9658 |              0.0123 |                  0.0393 |
| DecisionTree         |          0.8621 |          0.9556 |                   0.8578 |     0.8597 |             0.9553 |              0.6699 |                  0.0035 |
| LogisticRegression   |          0.8186 |          0.945  |                   0.7925 |     0.821  |             0.9413 |              0.2232 |                  0.0019 |

## 4. Explainability -- Most Important PIR Sensors
Method used: **SHAP_TreeExplainer_mean_abs_shap**

Top 10 most important features overall:

| feature   |   importance |
|:----------|-------------:|
| PIR_40    |      0.03463 |
| PIR_47    |      0.02341 |
| PIR_4     |      0.01941 |
| PIR_48    |      0.01623 |
| PIR_46    |      0.01595 |
| PIR_49    |      0.01373 |
| PIR_3     |      0.01073 |
| PIR_12    |      0.00874 |
| PIR_50    |      0.00827 |
| PIR_41    |      0.00776 |

## 5. PIR Sensor Reduction (Ablation)
|   n_pir_sensors |   macro_f1_mean |   macro_f1_std |   accuracy_mean |   mcc_mean |   mean_train_time_sec |   mean_inference_time_sec |
|----------------:|----------------:|---------------:|----------------:|-----------:|----------------------:|--------------------------:|
|               5 |         0.88759 |        0.01653 |         0.96418 |    0.88622 |               0.82133 |                   0.09082 |
|              10 |         0.90165 |        0.01505 |         0.96928 |    0.90237 |               1.15057 |                   0.12548 |
|              20 |         0.91157 |        0.01927 |         0.97267 |    0.91313 |               1.29507 |                   0.14196 |
|              30 |         0.91599 |        0.0198  |         0.97358 |    0.91589 |               1.45371 |                   0.14109 |
|              40 |         0.91566 |        0.02379 |         0.97384 |    0.91667 |               1.50832 |                   0.13143 |
|              55 |         0.91871 |        0.02389 |         0.97475 |    0.91958 |               1.59612 |                   0.13394 |

Using all 55 PIR sensors achieves Macro F1 = 0.9187. The best Macro F1 across all tested subset sizes was 0.9187 at K=55 sensors, indicating that a reduced sensor subset can retain competitive accuracy for lightweight deployment.

## 6. Computational Efficiency
| model                |   train_time_sec_full_data |   inference_time_ms_per_sample |   model_size_kb |   macro_f1_cv_mean |
|:---------------------|---------------------------:|-------------------------------:|----------------:|-------------------:|
| RandomForest         |                     2.1165 |                         0.0214 |         8438.07 |             0.9197 |
| ExtraTrees           |                     0.8436 |                         0.0215 |        27307.2  |             0.9189 |
| CatBoost             |                     7.2401 |                         0.0014 |          641.53 |             0.9162 |
| HistGradientBoosting |                     2.2792 |                         0.018  |         1673.43 |             0.9145 |
| XGBoost              |                     3.5311 |                         0.0051 |         1526.03 |             0.9118 |
| SVM                  |                     2.2545 |                         0.0956 |          420.66 |             0.9087 |
| MLP                  |                     1.0967 |                         0.0014 |          145.97 |             0.9038 |
| KNN                  |                     0.0145 |                         0.0168 |         3350.57 |             0.8892 |
| DecisionTree         |                     0.3596 |                         0.0004 |           17.47 |             0.8621 |
| LogisticRegression   |                     0.2771 |                         0.0007 |            4.63 |             0.8186 |

## 7. Temporal Robustness
_Full machine-readable data in `results/temporal_analysis.csv`; narrative below is generated from `results/temporal_holdout_report.txt`._
```
==============================================================================
TEMPORAL ROBUSTNESS / CHRONOLOGICAL STRESS ANALYSIS
==============================================================================

Per-date class distribution (rows=date, cols=Label):
               0    1    3
Date                      
2024-08-08   967    9  110
2024-09-08  4247  824  461
2024-10-08  1033    0    0

Final-chronological holdout window: last 15% of rows (1148 rows), covering dates: ['2024-09-08', '2024-10-08']
Classes present in this holdout window: [0]

RESULT TYPE: *** TEMPORAL STRESS ANALYSIS ONLY -- NOT a 3-class benchmark ***
  The final-chronological holdout window contains only 1 class(es): [0] (['Vacancy']).
  Reporting Macro F1/Precision/Recall here would be misleading (undefined or degenerate for absent classes), so only single-class-appropriate metrics are reported, and this is documented as a dataset LIMITATION: the tail of the collection period does not exercise all activity states, so a genuine end-of-timeline multiclass holdout is not currently possible with this dataset.
  Accuracy on holdout (single/degenerate class set) : 1.0000
  MCC on holdout                                    : 0.0000
  Actual holdout class counts: {0: 1148}

Principal evaluation remains the 5-fold StratifiedGroupKFold CV (results/model_cv_summary.csv), which IS a valid multiclass, temporally-leakage-controlled benchmark across the full date range.
==============================================================================
```

## 8. Limitations
- All data originates from a single canonical dataset file (`data/pirvision_office_dataset.csv`), i.e. a single data-collection run/session, so cross-session / cross-deployment generalization is untested.
- The final-chronological holdout is documented as a temporal stress analysis rather than a full multiclass benchmark whenever the tail of the collection period does not contain all three classes (see Section 7 above for the actual outcome observed in this run).
- Temperature_F contains sentinel/dropout values (==0) that were flagged, not imputed, to avoid introducing leakage or fabricated values.
- Hyperparameter search was intentionally small/manual (CPU-friendly, reproducible) rather than an exhaustive search; absolute performance ceilings may be somewhat higher with heavier tuning.
- All data originates from a single physical space/sensor rig; results may not transfer directly to a different room geometry or PIR sensor layout.

## 9. Practical Implications
- The PIR sensor reduction experiment (Section 5) directly informs lightweight edge deployment: fewer active PIR channels reduce wiring, power draw, and per-sample inference cost while the ablation table quantifies the accuracy trade-off explicitly.
- Group-aware, temporally-leakage-controlled cross-validation (StratifiedGroupKFold over 5-minute temporal buckets) gives a more realistic generalization estimate than a naive random row-wise split for this kind of densely, irregularly sampled sensor stream.
- Explainability results identify which physical PIR sensor positions and the ambient temperature channel drive predictions, supporting sensor-placement decisions in future smart-environment deployments.
