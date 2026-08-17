````markdown
# SkinXAI: Explainable Skin Lesion Classification with Deep CNNs

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC-blue)](https://www.isic-archive.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**SkinXAI** provides the implementation and experimental results for a comparative study of deep convolutional neural networks for three-class dermoscopic skin lesion classification with explainable artificial intelligence.

The repository is maintained by the **Visual Computing Center (VCC), University of Haripur**.

---

## Overview

This project evaluates six CNN architectures for classification of dermoscopic images into three categories:

- **Benign**
- **Benign Keratosis**
- **Malignant**

The evaluated architectures are:

1. DenseNet121
2. ConvNeXt-Tiny
3. ResNet152V2
4. EfficientNetV2-L
5. InceptionV3
6. Regularized Custom CNN

Five ImageNet-pretrained architectures are evaluated using a common two-phase transfer learning protocol, while the Custom CNN is trained from scratch and serves as a lightweight baseline.

The experiments were conducted under a constrained computing environment using an **NVIDIA Quadro M2000M GPU with 2.8 GB VRAM**.

---

## Dataset

The experiments use dermoscopic images curated from the **International Skin Imaging Collaboration (ISIC) Archive**.

The experimental dataset contains **30,000 images** distributed equally across the three target classes.

| Partition | Images |
|---|---:|
| Training | 24,000 |
| Validation | 3,000 |
| Test | 3,000 |
| **Total** | **30,000** |

The resulting split is approximately:

```text
80% Training
10% Validation
10% Test
````

The test set contains **1,000 unseen images per class**.

> **Note:** ISIC images are not redistributed through this repository. Users should obtain the original dermoscopic images from the ISIC Archive and prepare the dataset according to the experimental protocol described in the accompanying manuscript.

---

## Methodology

### Two-Phase Transfer Learning

The five pretrained CNN architectures follow a common two-phase training strategy.

#### Phase 1: Classification-Head Training

The ImageNet-pretrained backbone is frozen while the newly added classification head is trained for the first 20 epochs.

```text
Epochs:        1–20
Learning rate: 1 × 10^-4
Backbone:      Frozen
```

#### Phase 2: Selective Fine-Tuning

Selected deeper layers are subsequently unfrozen and fine-tuned using a substantially lower learning rate.

```text
Epochs:        21–50
Learning rate: 1 × 10^-6
Backbone:      Selectively unfrozen
```

This strategy allows task-specific adaptation while limiting large updates to the pretrained feature representations.

The **Custom CNN** does not use transfer learning and is trained from scratch for 50 epochs.

---

## Preprocessing and Data Augmentation

All images are resized to:

```text
224 × 224 × 3
```

Input preprocessing is adapted to the requirements of the corresponding ImageNet-pretrained architecture.

Training-time augmentation includes:

* random horizontal flipping;
* random vertical flipping;
* random rotation up to `0.2` radians;
* random zoom of approximately `±10%`.

Augmentation is performed on-the-fly during training.

---

## Training Configuration

The principal experimental settings are:

| Parameter                   | Setting                   |
| --------------------------- | ------------------------- |
| Optimizer                   | Adam                      |
| Loss                        | Categorical Cross-Entropy |
| Batch Size                  | 8                         |
| Phase 1 Learning Rate       | `1 × 10^-4`               |
| Phase 2 Learning Rate       | `1 × 10^-6`               |
| Phase 1                     | 20 epochs                 |
| Phase 2                     | 30 epochs                 |
| Total Training              | 50 epochs                 |
| Classification Head Dropout | 0.5                       |
| Input Resolution            | `224 × 224 × 3`           |
| GPU                         | NVIDIA Quadro M2000M      |
| Available VRAM              | 2.8 GB                    |

---

## Model Implementations

Training and evaluation scripts are available in the [`source_code/`](source_code/) directory.

| Architecture         | Implementation                                                             |
| -------------------- | -------------------------------------------------------------------------- |
| **Custom CNN**       | [`training_cnn.py`](source_code/training_cnn.py)                           |
| **ConvNeXt-Tiny**    | [`training_convext_tiny.py`](source_code/training_convext_tiny.py)         |
| **DenseNet121**      | [`training_densenet121.py`](source_code/training_densenet121.py)           |
| **EfficientNetV2-L** | [`training_efficientnetv2_l.py`](source_code/training_efficientnetv2_l.py) |
| **InceptionV3**      | [`training_inceptionv3.py`](source_code/training_inceptionv3.py)           |
| **ResNet152V2**      | [`training_resnet152v2.py`](source_code/training_resnet152v2.py)           |

---

## Evaluation

The architectures are evaluated using both training dynamics and held-out test performance.

The evaluation includes:

* classification accuracy;
* categorical cross-entropy loss;
* precision;
* recall;
* ROC curves;
* area under the ROC curve (AUC);
* precision-recall curves;
* average precision (AP);
* confusion matrices;
* Grad-CAM visualizations.

Training and validation trajectories are used to examine convergence behavior, while final classification analyses are performed using the held-out test partition.

---

## Experimental Results

Generated figures and evaluation outputs are available in the [`results/`](results/) directory.

### 1. Accuracy

```text
results/1_accuracy_curves/
```

Training and validation accuracy across the two-phase training process.

### 2. Loss

```text
results/2_loss_curves/
```

Training and validation categorical cross-entropy loss.

### 3. AUC Progression

```text
results/3_auc_curves/
```

Training and validation AUC across epochs.

### 4. Precision and Recall Progression

```text
results/4_precision_recall_curves/
```

Evolution of precision and recall during training.

### 5. ROC Analysis

```text
results/5_roc_curves/
```

One-vs-rest ROC curves and class-specific AUC values.

### 6. Confusion Matrices

```text
results/6_confusion_matrices/
```

Row-normalized confusion matrices evaluated on the held-out test set.

### 7. Precision-Recall Analysis

```text
results/7_pr_clinical_curves/
```

Class-wise precision-recall curves and corresponding average precision values.

### 8. Grad-CAM Explainability

```text
results/8_gradcam_panels/
```

Grad-CAM activation maps for representative Benign, Benign Keratosis, and Malignant samples across the evaluated CNN architectures.

Grad-CAM is used as a **qualitative interpretation method** to examine spatial regions contributing to model predictions. The resulting heatmaps should not be interpreted as direct evidence of clinical correctness.

---

## Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize the spatial regions contributing to individual CNN predictions.

For a target class (c), Grad-CAM computes importance weights from the gradients of the class score with respect to convolutional feature maps:

[
\alpha_k^c =
\frac{1}{Z}
\sum_i
\sum_j
\frac{\partial y^c}
{\partial A_{ij}^{k}}
]

The corresponding activation map is obtained as:

[
L_{\mathrm{Grad-CAM}}^c =
\mathrm{ReLU}
\left(
\sum_k
\alpha_k^c A^k
\right)
]

These activation maps provide a consistent qualitative basis for comparing spatial attention patterns across the six architectures.

---

## Key Observations

The experiments reveal architecture-specific differences in classification performance and optimization behavior.

* **EfficientNetV2-L** achieved the highest validation accuracy in the reported experiments.
* The pretrained CNN architectures generally outperformed the Custom CNN baseline.
* Different architectures exhibited different responses to selective fine-tuning after epoch 20.
* Class-wise analysis showed that **Benign Keratosis and Malignant lesions were more difficult to distinguish than Benign lesions**.
* Grad-CAM revealed differences in spatial activation patterns among the evaluated architectures.

The repository provides the associated curves, confusion matrices, and Grad-CAM visualizations for detailed inspection.

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/VCC-UOH/SkinXAI.git
cd SkinXAI
```

### Create a Virtual Environment

Using `venv`:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

Install the principal dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

If a `requirements.txt` file is provided, use:

```bash
pip install -r requirements.txt
```

---

## Run an Experiment

For example, to train DenseNet121:

```bash
python source_code/training_densenet121.py
```

To train ConvNeXt-Tiny:

```bash
python source_code/training_convext_tiny.py
```

Similarly, the remaining architectures can be executed using their corresponding scripts in [`source_code/`](source_code/).

> Dataset paths and output directories may need to be configured according to the local environment before execution.

---

## Repository Structure


SkinXAI/
│
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

## Demonstration and Releases

Supplementary execution material and repository releases can be accessed from:

**https://github.com/VCC-UOH/SkinXAI/releases**

Where available, releases provide archived versions of the implementation and associated supplementary material corresponding to the experimental study.

---

## Reproducibility

To facilitate reproducibility:

* all architectures use a common image resolution;
* the five pretrained architectures follow the same two-phase training strategy;
* the same batch size is used across experiments;
* common training and evaluation metrics are reported;
* the held-out test partition is not used for model training;
* experimental figures and model-specific outputs are provided in the repository.

Exact numerical results may vary slightly depending on TensorFlow version, GPU architecture, random initialization, and hardware configuration.

---

## Limitations

The current study has several limitations that should be considered when interpreting the results:

* evaluation is based on a single curated dermoscopic dataset;
* the experimental dataset is balanced across three lesion categories;
* external clinical validation has not yet been performed;
* Grad-CAM analysis is qualitative;
* demographic and acquisition-device variability are not explicitly evaluated;
* statistical significance testing across repeated independent runs is not included;
* inference-time and deployment benchmarking are outside the present evaluation.

These aspects are planned for future extensions of the project.

---

## Paper

This repository accompanies the manuscript:

> **Explainable Skin Lesion Classification: Comparative Evaluation and Visual Interpretation of Deep CNNs**

Authors:

**Sheryar Imran, Syed Abdul Hanan Hashmi, Dawar Khan, Wafa Bibi, Muhammad Hassan, and Cheng Wang**

If you use this repository in academic work, please cite the associated paper once its final bibliographic information becomes available.

---

## Citation

A BibTeX citation will be added after publication.

For the current version, the repository can be cited as:

```bibtex
@software{SkinXAI2026,
  title   = {SkinXAI: Explainable Skin Lesion Classification with Deep CNNs},
  author  = {Imran, Sheryar and
             Hashmi, Syed Abdul Hanan and
             Khan, Dawar and
             Bibi, Wafa and
             Hassan, Muhammad and
             Wang, Cheng},
  year    = {2026},
  url     = {https://github.com/VCC-UOH/SkinXAI}
}
```

---

## Contributors

* **Sheryar Imran**
* **Syed Abdul Hanan Hashmi**
* **Dawar Khan**
* **Wafa Bibi**
* **Muhammad Hassan**
* **Cheng Wang**

---

## Visual Computing Center

This repository is maintained under:

**Visual Computing Center (VCC)**
**University of Haripur, Pakistan**

GitHub organization:

**https://github.com/VCC-UOH**

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Disclaimer

This repository is intended for **research and educational purposes**. The models and visual explanations provided here have not undergone prospective clinical validation and should not be used as standalone tools for medical diagnosis or treatment decisions.
