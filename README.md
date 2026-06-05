# EdgeCaps-Pruned-Capsule-Networks-for-RF-Respiratory-Monitoring
EdgeCaps is an efficient Capsule Network-based contactless respiratory monitoring system designed for resource-constrained environments.

## Overview

A key goal of this work is enabling **real-time respiratory monitoring on portable and low-cost devices**.

To achieve this, we combine:

- Model compression (pruning + architecture reduction)
- Knowledge Distillation (KD)
- Lightweight Capsule Network design
- FLOPs and memory optimization

### Optimization Pipeline

Teacher Model → Student Model → Pruning  → Knowledge Distillation → Edge-Ready Model


## Repository Structure
This repository contains the codebase and experimental pipeline structured as:

- Hyperparameter search experiment
- Model training and SOTA comparison
- Edge deployment on Raspberry Pi-4

## Installation
pip install -r requirements.txt

## Usage
To train models: 
python -m scripts.run_experiment \
    --prune_ratio 0.3 \
    --temperature 5 \
    --alpha 0.7 \
    --lr 1e-4 \
    --batch_size 32 \
    --data_path ./data/Edgecaps_datasets

For the Raspberry Pi-deployment
 - Run the chroot.sh file to install all requirements and set the environment to support integration of radar and real-time data acquisition. 
 -

## Dataset
See `data/README.md`

## Citation
