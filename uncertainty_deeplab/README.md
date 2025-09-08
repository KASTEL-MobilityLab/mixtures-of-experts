# Uncertainty Evaluation for DeepLabv3+ on A2D2 and Cityscapes

This directory contains the evaluation code accompanying our STREAM@ICCV2025 paper [Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation](https://arxiv.org/abs/2509.04816).


## Overview

We provide evaluation scripts for semantic segmentation models on **A2D2**, focusing on:

- **Mixture of Experts (MoE)** with predictive entropy (PE), mutual information (MI), expert variance (EV), and gate entropy.
- **Ensembles** of DeepLabv3+ experts (urban + highway).
- **MC Dropout** applied to DeepLabv3+ baseline models.

Evaluations cover:
- **Clean data** (severity 0)
- **Corrupted data** (severity levels 1–5)
- **Out-of-distribution (ambiguous A2D2 subset)**
- **Robustness under systematic data shift**

Metrics include **mIoU**, **ECE**, **MCE**, **Brier score**, **NLL**, and **conditional correctness** (p(acc|certain), p(uncertain|inaccurate), PAvPU).



## Directory Structure

### Ensemble
- `eval_ensemble_all_severities.py` – Evaluate DeepLabV3+ ensemble across all corruption severities.
- `eval_ensemble_clean.py` – Evaluate ensemble on clean (severity 0) data.

### MC Dropout
- `eval_mc_dropout_all_severities.py` – Evaluate MC Dropout across all severities.
- `eval_mc_dropout_clean.py` – Evaluate MC Dropout on clean data.
- `eval_mc_dropout_miou_clean.py` – Compute mIoU for MC Dropout on clean data only.
- `eval_mc_dropout_miou_all_severities.py` – Compute mIoU for MC Dropout across all severities.

### Mixture of Experts (MoE) 
- `eval_moe_miou_all_severities.py` – Compute mIoU for MoE across all severities.
- `eval_moe_stacked_all_severities.py` – Robustness evaluation for MoE models.

### Uncertainty Extraction for MoEs
- `eval_moe_weighted_clean.py` – Uncertainty evaluation with weighted expert contributions.
- `eval_moe_stacked_clean.py` – Uncertainty evaluation including MoE output fusion.


## Usage Examples

## Ensemble
`python eval_ensemble_clean.py params/params_ensemble.yaml`

`python eval_ensemble_all_severities.py params/params_ensemble.yaml`

## MC Dropout – all severities + uncertainty quantification (PE/MI + PAvPU, ECE/MCE/Brier/NLL)
Single rate (rate, MC samples, out dir)

`python eval_mc_dropout_all_severities.py params/params_mc_dropout.yaml 0.10 2 ./perturbation/mc_dropout`

Sweep rates (out dir, MC samples optional)

`python eval_mc_dropout_all_severities.py params/params_mc_dropout.yaml --severity-sweep ./perturbation/mc_dropout 2`

## MC Dropout – clean (severity 0) + UQ
Single rate (rate, MC samples)

`python eval_mc_dropout_clean.py params/params_mc_dropout.yaml 0.10 2`

Sweep rates (out dir, MC samples optional)

`python eval_mc_dropout_clean.py params/params_mc_dropout.yaml --sweep ./mc_dropout_ambiguous 2`

## MC Dropout mIoU only – all severities
Single rate (rate, MC samples, out dir)

`python eval_mc_dropout_miou_all_severities.py params/params_mc_dropout.yaml --severity 0.10 2 ./single_mc_dropout_severity`

Sweep rates (out dir, MC samples optional)

`python eval_mc_dropout_miou_all_severities.py params/params_mc_dropout.yaml --severity-sweep ./mc_dropout_severity 2`

Legacy single clean run (no corruptions)

`python eval_mc_dropout_miou_all_severities.py params/params_mc_dropout.yaml 0.10 2`

## MC Dropout mIoU only – clean (severity 0)

Single rate (rate, MC samples)

`python eval_mc_dropout_miou_clean.py params/params_mc_dropout.yaml 0.10 2`

Sweep rates (out dir, MC samples optional)

`python eval_mc_dropout_miou_clean.py params/params_mc_dropout.yaml --sweep ./results/mc_dropout_all_miou 2`

## MoE mIoU across all severities
`python eval_moe_miou_all_severities.py params/params_moe.yaml`

## MoE stacked – all severities (PE/MI/EVU + PAvPU)
`python moe_stacked_all_severities.py`

## MoE stacked – clean (severity 0)
`python moe_stacked_clean.py`

## MoE weighted (gate-weighted) – clean (severity 0)
`python moe_weighted_clean.py`


# Citation

If you find this code useful for your research, please cite our paper:

```latex
@inproceedings{pavlitska2025extracting,
  author = {Svetlana Pavlitska and 
          Beyza Keskin and
          Alwin Faßbender and 
          Christian Hubschneider and
          J. Marius Zöllner},
  title = {Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation},
  booktitle = {International Conference on Computer Vision (ICCV) - Workshops},
  year      = {2025}
}
```



