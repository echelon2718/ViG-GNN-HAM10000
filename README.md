# Leveraging Spatial and Semantic Feature Extraction for Skin Cancer Diagnosis with Capsule Networks and Graph Neural Networks

## Overview

This repository contains the implementation and supplementary materials for the [paper](https://arxiv.org/abs/2403.12009) titled "Leveraging Spatial and Semantic Feature Extraction for Skin Cancer Diagnosis with Capsule Networks and Graph Neural Networks."

## Abstract

Skin lesion classification presents challenges due to intricate spatial and semantic features and imbalanced datasets. This paper introduces a hybrid model combining Graph Neural Networks (GNNs) and Capsule Networks to enhance classification performance. Our model, applied to the MNIST:HAM10000 dataset, achieved superior accuracy compared to benchmarks, demonstrating significant improvements in diagnosing skin cancer.

## Authors

- K. P. Santoso
- R. V. H. Ginardi
- R. A. Sastrowardoyo
- F. A. Madany

## Key Features

- Integration of GNNs and Capsule Networks
- Application to MNIST:HAM10000 skin lesion dataset
- Achieved accuracy: 95.52%

## Results

Our model outperformed established benchmarks:
- GoogLeNet: 83.94%
- InceptionV3: 86.82%
- MobileNet V3: 89.87%
- EfficientNet-B7: 92.07%
- ResNet18: 92.22%
- ResNet34: 91.90%
- ViT-Base: 73.70%
- IRv2-SA: 93.47%

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/skin-cancer-diagnosis.git
cd skin-cancer-diagnosis
pip install -r requirements.txt
