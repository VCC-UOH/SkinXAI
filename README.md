# 🔬 Comparative Analysis of Deep Learning Models for Skin Lesion Classification with Explainable AI (XAI)

![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-FF6F00?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![Dataset](https://img.shields.io/badge/Dataset-30%2C000_ISIC_Images-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎬 Execution & Demonstration Video

> Video proof showcasing training pipeline, execution results, (Grad-CAM) evaluation:

<div align="center">
  <video src="sha256:bbb97dc7908190277961ae1089abd6f6afab9b8e8cebf2b76f23cf2b15d4ad36" controls="controls" width="100%">
    Your browser does not support playing video directly.
  </video>
</div>

---

## 📌 Project Overview
This repository contains the official implementation, experimental visual outputs, and source scripts for automated 3-class skin lesion classification (`Benign`, `Benign Keratosis`, `Malignant`). 

The study evaluates **6 deep learning architectures** trained on a balanced dataset of **30,000 dermatoscopic images** curated from the ISIC Archive ($80 : 10 : 10$ split across $24,000$ training, $3,000$ validation, and $3,000$ unseen test images).

---

## 💻 Source Code & Model Architectures

All training and evaluation scripts are available in the [`source_code/`](source_code/) directory:

| Architecture | Python Script File | Pipeline Description |
| :--- | :--- | :--- |
| **Custom CNN** | [`training_cnn.py`](source_code/training_cnn.py) | Custom regularized CNN baseline |
| **ConvNeXt-Tiny** | [`training_convext_tiny.py`](source_code/training_convext_tiny.py) | Modernized pure-convolutional architecture |
| **DenseNet121** | [`training_densenet121.py`](source_code/training_densenet121.py) | Dense feature-reuse network |
| **EfficientNetV2-L** | [`training_efficientnetv2_l.py`](source_code/training_efficientnetv2_l.py) | Compound-scaled ConvNet |
| **InceptionV3** | [`training_inceptionv3.py`](source_code/training_inceptionv3.py) | Multi-scale factorized convolutions |
| **ResNet152V2** | [`training_resnet152v2.py`](source_code/training_resnet152v2.py) | Deep residual skip-connection network |

---

## 📊 Experimental Results & Visual Proofs

All raw high-resolution output figures are stored in the [`results/`](results/) folder across 8 evaluation categories:

<details open>
<summary><b>📈 1. Epoch-wise Accuracy Trajectories</b></summary>
<br>
📁 <i>Files stored in <code>results/1_accuracy_curves/</code></i>
</details>

<details>
<summary><b>📉 2. Categorical Cross-Entropy Loss Curves</b></summary>
<br>
📁 <i>Files stored in <code>results/2_loss_curves/</code></i>
</details>

<details>
<summary><b>🎯 3. Area Under Curve (AUC) Performance</b></summary>
<br>
📁 <i>Files stored in <code>results/3_auc_curves/</code></i>
</details>

<details>
<summary><b>🔄 4. Precision-Recall Curves</b></summary>
<br>
📁 <i>Files stored in <code>results/4_precision_recall_curves/</code></i>
</details>

<details>
<summary><b>⚡ 5. Receiver Operating Characteristic (ROC) Curves</b></summary>
<br>
📁 <i>Files stored in <code>results/5_roc_curves/</code></i>
</details>

<details open>
<summary><b>🧩 6. Confusion Matrices (3,000 Unseen Test Samples)</b></summary>
<br>
📁 <i>Files stored in <code>results/6_confusion_matrices/</code></i>
</details>

<details>
<summary><b>🏥 7. Clinical PR Curves</b></summary>
<br>
📁 <i>Files stored in <code>results/7_pr_clinical_curves/</code></i>
</details>

<details open>
<summary><b>🔍 8. Grad-CAM Explainable AI (XAI) Heatmaps</b></summary>
<br>
📁 <i>Files stored in <code>results/8_gradcam_panels/</code></i>
</details>

---

## 🚀 Environment Requirements
To run any script in `source_code/`:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python