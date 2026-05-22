# EdgeCaps-Pruned-Capsule-Networks-for-RF-Respiratory-Monitoring
EdgeCaps is an efficient Capsule Networks based contactless respiratory monitoring system designed for resource constrained environments.

## Overview
This repository provides the implementation of **EdgeCaps**, a lightweight Capsule Network designed for efficient respiratory anomaly detection using Ultra-Wideband (UWB) radar signals. The proposed framework targets deployment on resource-constrained edge devices by combining model compression and knowledge transfer techniques.

## Key Contributions
- **Multi-stage compression pipeline**: architecture reduction, structured and unstructured pruning, and knowledge distillation  
- **Significant efficiency gains**: ~95% reduction in FLOPs and model size  
- **Edge deployment**: validated on Raspberry Pi for real-time respiratory monitoring  
- **Comprehensive evaluation**: comparison with state-of-the-art lightweight architectures (MobileNetV3, ShuffleNet, SqueezeNet)

## Repository Structure
This repository contains the codebase and experimental pipeline used to develop and evaluate the EdgeCaps framework. The proposed approach leverages Capsule Networks to effectively capture the inherent temporal and spatial variations in UWB radar signals for respiratory classification. Model efficiency is further enhanced through pruning and knowledge distillation.

## Installation
pip install -r requirements.txt

## Usage


## Dataset
See `data/README.md`

## Citation
(Add your paper)