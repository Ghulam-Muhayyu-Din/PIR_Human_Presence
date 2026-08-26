# Human Presence Detection from Multichannel PIR Sensors

A reproducible research pipeline for three class human presence classification using a 55 channel passive infrared sensor array. The study evaluates classical and ensemble machine learning models under temporal group aware cross validation, identifies influential PIR channels using SHAP, and quantifies performance as the number of active PIR channels is reduced.

The repository accompanies the research manuscript:

**Human Presence Detection from Multichannel PIR Sensors**

## Research Summary

The study uses a single office dataset derived from the PIRvision collection available through the UCI Machine Learning Repository.

The primary experiment uses:

• 7,651 observations
• 55 PIR channels
• Three target classes
• Five fold StratifiedGroupKFold cross validation
• 300 second temporal grouping
• Macro F1 as the primary model selection metric
• Ten machine learning models
• SHAP based feature attribution
• Fold specific PIR sensor ablation
• Desktop CPU computational analysis

The primary feature space contains **PIR channels only**. `Temperature_F` and the derived temperature zero indicator are excluded from the primary model because the data quality audit showed that the zero temperature value was strongly associated with the other activity or motion class and could act as a dataset specific shortcut.

## Frozen Primary Result

The selected model is **Random Forest**.

| Metric | Result |
|---|---:|
| Mean Macro F1 | 0.919684 |
| Mean Accuracy | 0.974884 |
| Mean MCC | 0.919939 |
| Evaluation | 5 fold StratifiedGroupKFold |
| Temporal grouping | 300 seconds |
| Primary features | 55 PIR channels |

Extra Trees achieved a very similar Macro F1 of 0.918890 and a slightly higher mean MCC of 0.921137. Therefore, Random Forest is selected specifically because Macro F1 was the predefined primary criterion, not because it is universally superior.

## Target Classes

The original dataset labels are preserved:

| Original Label | Class |
|---:|---|
| 0 | Vacancy |
| 1 | Stationary human presence |
| 3 | Other activity or motion |

For model fitting, the pipeline internally maps:

```text
0 → 0
1 → 1
3 → 2
```

The original labels and class meanings are restored in exported reports and predictions.

## Dataset

Place the canonical dataset at:

```text
data/pirvision_office_dataset.csv
```

The project is designed around one canonical dataset file. The pipeline does not require a duplicate raw dataset and does not compare multiple source files.

The local study file contains:

```text
7,651 rows
59 columns
55 PIR channels
Date
Time
Label
Temperature_F
```

The public dataset record is:

PIRvision_FoG_presence_detection, UCI Machine Learning Repository  
DOI: `10.24432/C56W5M`

https://doi.org/10.24432/C56W5M

Before redistributing the CSV through a public repository, verify the dataset's current redistribution conditions. This repository can instead provide the dataset citation and instructions for obtaining the source file.

## Project Structure

```text
PIR_Human_Presence/
│
├── data/
│   └── pirvision_office_dataset.csv
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_inspection.py
│   ├── feature_identification.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── cross_validation.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── sensor_ablation.py
│   ├── temporal_analysis.py
│   ├── figures.py
│   ├── figure_qc.py
│   └── report_generation.py
│
├── results/
├── figures/
├── models/
├── logs/
│
├── run_all.py
├── regenerate_frozen_figures.py
├── requirements.txt
└── README.md
```

## Pipeline

Run the complete pipeline through:

```bash
python -u run_all.py
```

The stages are:

```text
Single Dataset Validation
        ↓
Dataset Audit
        ↓
Feature Identification
        ↓
PIR Only Preprocessing
        ↓
Temporal Group Construction
        ↓
5 Fold StratifiedGroupKFold
        ↓
10 Model Comparison
        ↓
Random Forest Selection
        ↓
Out of Fold Evaluation
        ↓
SHAP Feature Attribution
        ↓
Fold Specific Sensor Ablation
        ↓
Computational Efficiency
        ↓
Temporal Stress Analysis
        ↓
Publication Figures and Quality Control
        ↓
Scientific Results Summary
```

## Model Comparison

Ten models are evaluated under the same primary protocol:

```text
Random Forest
Extra Trees
CatBoost
HistGradientBoosting
XGBoost
Support Vector Machine
Multilayer Perceptron
K Nearest Neighbors
Decision Tree
Logistic Regression
```

The search uses small CPU friendly parameter grids rather than an exhaustive hyperparameter optimization procedure.

## Cross Validation

The principal evaluation uses:

```text
StratifiedGroupKFold
Number of folds = 5
Temporal group width = 300 seconds
```

Date and time are used only to construct temporal groups. They are not passed to the classifiers as model features.

All observations within the same temporal group are assigned to the same fold. This reduces the risk that highly related adjacent observations cross the training and validation boundary.

All preprocessing that requires fitting is performed inside the model pipeline for the corresponding training fold.

## Explainability

The selected Random Forest is interpreted with:

```text
SHAP TreeExplainer
```

Important PIR channels in the frozen experiment include:

```text
PIR_40
PIR_47
PIR_4
PIR_48
PIR_46
PIR_49
PIR_3
PIR_12
PIR_50
PIR_41
```

These values represent model attribution. They are not interpreted as causal physical effects of individual PIR sensors.

## PIR Sensor Reduction

A fold specific sensor ablation experiment evaluates:

```text
5 sensors
10 sensors
20 sensors
30 sensors
40 sensors
55 sensors
```

Frozen Macro F1 results:

| PIR Channels | Macro F1 |
|---:|---:|
| 5 | 0.887591 |
| 10 | 0.901646 |
| 20 | 0.911575 |
| 30 | 0.915992 |
| 40 | 0.915664 |
| 55 | 0.918715 |

For the evaluated dataset and protocol, the 30 channel configuration retains approximately **99.7 percent** of the full 55 channel configuration's Macro F1.

The sensor ranking is derived separately from each training fold before evaluation on its corresponding validation fold. This avoids using validation observations to decide which sensors should be selected.

The 30 channel result is dataset specific and should not be interpreted as a universal hardware optimum.

## Temperature Data Quality Decision

An earlier PIR plus temperature experiment achieved approximately:

```text
Macro F1 ≈ 0.9897
```

The data quality audit found that:

```text
Temperature_F = 0
```

occurred in all 571 observations of the other activity or motion class.

A temperature zero indicator also became a dominant explanatory feature.

Because this pattern could provide a class correlated shortcut rather than genuine generalizable temperature information, temperature was excluded from the primary experiment.

The earlier 0.9897 result is therefore **not** reported as the main research result.

## Computational Analysis

For each model, the pipeline records:

• Cross validation fit time
• Cross validation prediction time
• Full dataset training time
• Per sample inference time
• Serialized model size

These measurements were obtained on the desktop CPU environment used for the frozen pipeline.

They are **not** embedded device, edge device, or real time deployment measurements.

## Figures

The figure generation code produces the manuscript figures directly from the frozen project outputs.

| Figure | File | Purpose |
|---|---|---|
| 1 | `figure_01_workflow` | Research workflow |
| 2 | `figure_02_class_distribution` | Target class distribution |
| 3 | `figure_03_model_comparison` | Model comparison using Macro F1 and MCC |
| 4 | `figure_04_confusion_matrix` | Out of fold confusion matrix |
| 5 | `figure_05_shap_feature_importance` | PIR channel attribution |
| 6 | `figure_06_pir_sensor_reduction` | Performance versus number of PIR channels |
| 7 | `figure_07_performance_vs_cost` | Performance and computational trade off |
| 8 | `figure_08_correlation_heatmap` | PIR channel correlation |
| 9 | `figure_09_per_class_performance` | Precision, recall, and F1 by class |
| 10 | `figure_10_temporal_class_distribution` | Temporal class distribution and stress analysis |

Figures are generated programmatically and exported as:

```text
PNG, 600 DPI
PDF
SVG
```

The figure quality control stage checks generated outputs and writes:

```text
results/figure_audit.csv
```

For manuscript submission, the original frozen project figures are the authoritative visual outputs.

## Reproducibility

The pipeline uses a global seed of:

```text
42
```

The configuration is stored in:

```text
configs/config.yaml
```

The principal results are written to:

```text
results/model_cv_summary.csv
results/model_fold_results.csv
results/classification_report_best.csv
results/oof_predictions_best.csv
results/feature_importance.csv
results/ablation_results.csv
results/computational_efficiency.csv
results/temporal_analysis.csv
results/final_model_parameters.json
```

Models are saved under:

```text
models/
```

Logs are saved under:

```text
logs/
```

## PyCharm Setup

Recommended Python version:

```text
Python 3.11
```

Create the environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.venv\Scriptsctivate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the complete experiment:

```powershell
python -u run_all.py
```

The `-u` option keeps console output unbuffered so model training and pipeline progress are immediately visible in PyCharm.

## Regenerating Figures Without Retraining

Once the experimental results have been frozen, use:

```powershell
python regenerate_frozen_figures.py
```

This regenerates the figures from the saved experiment outputs and does not retrain the models.

## Research Integrity

This repository is intended to support reproducibility and transparent reporting.

The reported primary experiment uses PIR features only.

The earlier temperature assisted result is retained only as a documented data quality and leakage diagnosis.

No result should be interpreted as a universal deployment benchmark.

The current evidence is based on one local office dataset file and one data collection context. External validation across other buildings, sessions, sensor layouts, and deployments remains future work.

## Manuscript

The accompanying manuscript is:

**Human Presence Detection from Multichannel PIR Sensors**

The repository contains the computational evidence underlying the manuscript, including the model comparison, out of fold evaluation, SHAP analysis, sensor reduction study, computational analysis, and final figures.

## Citation

If you use this repository or the experimental pipeline in academic work, please cite the accompanying manuscript:

```text
Din, G. M. Human Presence Detection from Multichannel PIR Sensors.
Ubiquitous Technology Journal, CrossLink Studies.
```

The final bibliographic information should be updated after publication.

## Dataset Citation

Please cite the original PIRvision dataset:

```text
M. Emad-ud-din, "PIRvision_FoG_presence_detection,"
UCI Machine Learning Repository, 2023,
doi: 10.24432/C56W5M.
```

## License

Add the repository license that matches the licensing conditions you intend to apply to your original code and documentation. Do not apply a license to third party dataset material unless its terms permit redistribution under that license.
