# Frozen PIR Human Presence Experiment

This package contains the frozen experimental outputs after the final PIR only experiment.

## Frozen experimental configuration

Primary inputs: 55 PIR sensors only

Temperature was excluded from the primary model input.

Evaluation: 5 fold StratifiedGroupKFold with temporal groups of 300 seconds.

Primary selection metric: Macro F1.

Selected model: Random Forest.

Cross validation Macro F1: 0.919684.

Cross validation accuracy: 0.974905.

Out of fold MCC: 0.920001.

## Class wise out of fold performance

Vacancy: Precision 0.992349, Recall 0.996638, F1 0.994489.

Stationary human presence: Precision 0.888889, Recall 0.912365, F1 0.900474.

Other activity and motion: Precision 0.906130, Recall 0.828371, F1 0.865508.

## Important reproducibility rule

Do not rerun model training when preparing the manuscript. The result tables in this package are frozen. Figure regeneration is performed only from the frozen result tables and the canonical dataset.

## Primary scientific point

The full 55 sensor configuration achieved Macro F1 0.918715 in the fold specific sensor ablation analysis, while 30 sensors achieved 0.915992.
