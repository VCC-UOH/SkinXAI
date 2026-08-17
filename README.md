# SkinXAI: Explainable AI for Skin Lesion Classification

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

## 💻 Source Code & Model Architectures

Training and evaluation scripts for all six architectures are available in the [`source_code/`](source_code/) directory.

| Architecture | Python Script | Description |
|---|---|---|
| **Custom CNN** | [`training_cnn.py`](source_code/training_cnn.py) | Regularized CNN trained from scratch |
| **ConvNeXt-Tiny** | [`training_convext_tiny.py`](source_code/training_convext_tiny.py) | Modern convolutional architecture |
| **DenseNet121** | [`training_densenet121.py`](source_code/training_densenet121.py) | Dense feature-reuse architecture |
| **EfficientNetV2-L** | [`training_efficientnetv2_l.py`](source_code/training_efficientnetv2_l.py) | Efficient compound-scaled architecture |
| **InceptionV3** | [`training_inceptionv3.py`](source_code/training_inceptionv3.py) | Multi-scale convolutional architecture |
| **ResNet152V2** | [`training_resnet152v2.py`](source_code/training_resnet152v2.py) | Deep residual architecture |

---

## 🛠️ Training Pipeline

### Two-Phase Transfer Learning

Five ImageNet-pretrained architectures are trained using a common two-phase transfer learning strategy:

- **Phase 1 — Classification Head Training:** The pretrained backbone is frozen and the classification head is trained for the first 20 epochs using a learning rate of `1e-4`.
- **Phase 2 — Selective Fine-Tuning:** Selected deeper layers are unfrozen and fine-tuned from epochs 21–50 using a reduced learning rate of `1e-6`.

The **Custom CNN** is trained from scratch for 50 epochs and serves as a lightweight baseline.

### Data Pipeline

- Input resolution: `224 × 224 × 3`
- Batch size: `8`
- Optimizer: `Adam`
- Loss: `Categorical Cross-Entropy`
- Real-time data augmentation
- Horizontal and vertical flipping
- Random rotation and zoom
- `tf.data` prefetching and parallel mapping

### Training and Monitoring

- CSV-based training logs
- Model checkpointing based on validation loss
- Early stopping with best-weight restoration
- Training and validation metric tracking
- Grad-CAM generation for qualitative model interpretation

The experiments were conducted under a constrained GPU environment with approximately **2.8 GB VRAM**.

---

## 📊 Experimental Results

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

## 🔍 Explainable AI

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to visualize image regions contributing to model predictions.

Grad-CAM heatmaps are generated for representative samples from the three lesion categories:

- Benign
- Benign Keratosis
- Malignant

The visualizations enable qualitative comparison of spatial attention patterns across the evaluated architectures.

Grad-CAM is used as a qualitative interpretation method and should not be considered direct evidence of clinical correctness.

---

## 🚀 Quick Start

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

## 📁 Repository Structure

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

## 📄 Paper

This repository accompanies the manuscript:

**Explainable Skin Lesion Classification: Comparative Evaluation and Visual Interpretation of Deep CNNs**

**Authors:** Sheryar Imran, Syed Abdul Hanan Hashmi, Dawar Khan, Wafa Bibi, Muhammad Hassan, and Cheng Wang.

If you find this work useful, please consider citing it. The citation will be updated with the final publication details once available.

```bibtex
@article{imran2026skinxai,
  title  = {Explainable Skin Lesion Classification: Comparative Evaluation and Visual Interpretation of Deep CNNs},
  author = {Imran, Sheryar and Hashmi, Syed Abdul Hanan and Khan, Dawar and Bibi, Wafa and Hassan, Muhammad and Wang, Cheng},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Disclaimer

This repository is intended for **research and educational purposes**. The models have not undergone prospective clinical validation and should not be used as standalone medical diagnostic tools.