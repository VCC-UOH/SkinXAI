# SkinXAI: Explainable Skin Lesion Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC-blue)](https://www.isic-archive.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **SkinXAI**, a comparative study of deep CNN architectures for three-class skin lesion classification with Grad-CAM-based explainability.

---

## Overview

We evaluate six CNN architectures for dermoscopic skin lesion classification into three categories: **Benign**, **Benign Keratosis**, and **Malignant**.

- DenseNet121
- ConvNeXt-Tiny
- ResNet152V2
- EfficientNetV2-L
- InceptionV3
- Custom CNN

The experiments use **30,000 dermoscopic images** curated from the ISIC Archive with an 80:10:10 train/validation/test split.

| Split | Images |
|---|---:|
| Training | 24,000 |
| Validation | 3,000 |
| Test | 3,000 |
| **Total** | **30,000** |

---

## Methodology

Five ImageNet-pretrained architectures use a two-phase transfer learning strategy.

### Phase 1: Classification Head Training

- Frozen pretrained backbone
- Epochs 1–20
- Learning rate: `1e-4`

### Phase 2: Selective Fine-Tuning

- Selected deeper layers unfrozen
- Epochs 21–50
- Learning rate: `1e-6`

The **Custom CNN** is trained from scratch as a lightweight baseline.

All experiments use `224 × 224` images, categorical cross-entropy loss, Adam optimization, and a batch size of 8.

**Grad-CAM** is used to provide qualitative visual interpretation of model predictions and compare spatial attention patterns across architectures.

---

## Source Code

Training and evaluation scripts are available in the [`source_code/`](source_code/) directory.

| Architecture | Script |
|---|---|
| Custom CNN | [`training_cnn.py`](source_code/training_cnn.py) |
| ConvNeXt-Tiny | [`training_convext_tiny.py`](source_code/training_convext_tiny.py) |
| DenseNet121 | [`training_densenet121.py`](source_code/training_densenet121.py) |
| EfficientNetV2-L | [`training_efficientnetv2_l.py`](source_code/training_efficientnetv2_l.py) |
| InceptionV3 | [`training_inceptionv3.py`](source_code/training_inceptionv3.py) |
| ResNet152V2 | [`training_resnet152v2.py`](source_code/training_resnet152v2.py) |

---

## Experimental Results

Experimental outputs are available in the [`results/`](results/) directory.

The repository includes:

- Training and validation accuracy curves
- Loss curves
- AUC analysis
- Precision and recall analysis
- ROC curves
- Precision-Recall curves
- Confusion matrices
- Grad-CAM visualizations

These results provide a common basis for comparing classification performance, convergence behavior, and visual explanations across the six architectures.

---

## Quick Start

Clone the repository:

```bash
git clone https://github.com/VCC-UOH/SkinXAI.git
cd SkinXAI
```

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

Run a model, for example DenseNet121:

```bash
python source_code/training_densenet121.py
```

Dataset paths should be configured according to the local environment before training.

---

## Repository Structure

```text
SkinXAI/
├── source_code/
│   ├── training_cnn.py
│   ├── training_convext_tiny.py
│   ├── training_densenet121.py
│   ├── training_efficientnetv2_l.py
│   ├── training_inceptionv3.py
│   └── training_resnet152v2.py
│
├── results/
│   ├── 1_accuracy_curves/
│   ├── 2_loss_curves/
│   ├── 3_auc_curves/
│   ├── 4_precision_recall_curves/
│   ├── 5_roc_curves/
│   ├── 6_confusion_matrices/
│   ├── 7_pr_clinical_curves/
│   └── 8_gradcam_panels/
│
├── LICENSE
└── README.md
```

---

## Dataset

The dermoscopic images used in this study are derived from the **International Skin Imaging Collaboration (ISIC) Archive**.

The original images are not redistributed through this repository. Please obtain the dataset from the official ISIC source.

---

## Paper

This repository accompanies the manuscript:

**Explainable Skin Lesion Classification: Comparative Evaluation and Visual Interpretation of Deep CNNs**

**Authors:** Sheryar Imran, Syed Abdul Hanan Hashmi, Dawar Khan, Wafa Bibi, Muhammad Hassan, and Cheng Wang.

If you find this work useful, please consider citing it. The citation will be updated with the final publication details once available.

```bibtex
@article{imran2026skinxai,
  title  = {Explainable Skin Lesion Classification: Comparative Evaluation and Visual Interpretation of Deep CNNs},
  author = {Imran, Sheryar and Hashmi, Syed Abdul Hanan and Khan, Dawar and Bibi, Wafa and Hassan, Muhammad and Wang, Cheng},
  year   = {2026},
  note   = {Manuscript submitted/to be submitted}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Disclaimer

This repository is intended for **research and educational purposes**. The models have not undergone prospective clinical validation and should not be used as standalone medical diagnostic tools.