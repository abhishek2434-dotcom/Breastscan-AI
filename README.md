%%writefile README.md
# BreastScan-AI: Breast Ultrasound Tumor Classification

![BreastScan-AI Logo](https://raw.githubusercontent.com/EchoScan-AI/EchoScan-AI/main/docs/logo.png) <!-- Placeholder for a project logo if available -->

## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Overview](#2-project-overview)
3. [Features](#3-features)
4. [Methodology](#4-methodology)
5. [Installation & Setup](#5-installation--setup)
6. [Usage](#6-usage)
7. [Model Performance](#7-model-performance)
8. [Future Work](#8-future-work)
9. [Contributing](#9-contributing)
10. [License](#10-license)
11. [Contact](#11-contact)

---

## 1. Introduction

EchoScan-AI is an open-source project dedicated to advancing the early and accurate detection of breast cancer through automated analysis of ultrasound images. Leveraging state-of-the-art machine learning and deep learning techniques, this project aims to provide a robust and interpretable system for classifying breast tumors as normal, benign, or malignant.

Early diagnosis is critical for effective breast cancer treatment. While ultrasound is a widely used and non-invasive imaging modality, its interpretation can be subjective. EchoScan-AI seeks to augment the diagnostic process, reduce inter-observer variability, and ultimately improve patient outcomes by offering an objective and highly accurate computational tool.

## 2. Project Overview

This repository contains the full pipeline for breast ultrasound tumor classification, encompassing:
*   **Data Preparation:** Scripts for organizing and preprocessing the Breast Ultrasound Images (BUSI) dataset.
*   **Feature Extraction:** Modules for extracting classical handcrafted features (texture, shape).
*   **Classical Machine Learning Models:** Implementations of Random Forest, Support Vector Machines (SVM), and XGBoost.
*   **Deep Learning Models:** Adaptations and training pipelines for YOLOv11 (Nano and Small variants) and Vision Transformers (ViT) for classification.
*   **Ensemble Modeling:** Conceptual framework for combining multiple models to enhance predictive accuracy.
*   **Interpretability Tools:** Integration of techniques like Grad-CAM for model explainability.
*   **Comprehensive Documentation:** Detailed reports and scientific paper outlines.

## 3. Features

*   **Multi-Model Approach:** Evaluation of diverse classical and deep learning algorithms.
*   **Automated Data Pipeline:** Streamlined processes for dataset handling and model training.
*   **High Accuracy:** Deep learning models, especially ViT and YOLO variants, demonstrate superior performance.
*   **Interpretability (XAI):** Tools to visualize and understand model decisions.
*   **Scalable:** Designed to be adaptable for larger datasets and different imaging modalities.
*   **Research-Oriented:** Focus on comparative analysis and future research directions.

## 4. Methodology

The project utilizes the Breast Ultrasound Images (BUSI) dataset, comprising normal, benign, and malignant cases. The core methodology involves:

*   **Classical ML:** Extracting texture (statistical, spectral, GLCM) and shape features, followed by classification using Random Forest, SVM, and hyperparameter-tuned XGBoost.
*   **Deep Learning:** Fine-tuning pre-trained YOLOv11 (Nano and Small) models and a Vision Transformer (`google/vit-base-patch16-224`) on the BUSI dataset for image classification.
*   **Evaluation:** Rigorous performance assessment using accuracy, confusion matrices, and cross-validation.

## 5. Installation & Setup

To set up the EchoScan-AI project locally or in a Colab environment:

### 5.1. Clone the Repository
```bash
git clone https://github.com/EchoScan-AI/EchoScan-AI.git
cd EchoScan-AI
```

### 5.2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
**(Note: `requirements.txt` will be generated or available soon, containing `pandas`, `numpy`, `scikit-learn`, `xgboost`, `ultralytics`, `transformers`, `torch`, `torchvision`, `Pillow`, `scipy`, `scikit-image`, `matplotlib`, `seaborn`, `grad-cam`, `timm` etc.)**

### 5.3. Dataset Download
The project uses the Breast Ultrasound Images (BUSI) dataset available on Kaggle. You will need a Kaggle API token.

1.  **Download `kaggle.json`:** Follow instructions [here](https://www.kaggle.com/docs/api#authentication).
2.  **Place `kaggle.json`:** Move it to `~/.kaggle/` and set permissions: `chmod 600 ~/.kaggle/kaggle.json`.
3.  **Download & Unzip:**
    ```bash
    kaggle datasets download -d abdullahalhammad/breast-ultrasound-images-for-breast-cancer
    unzip -o breast-ultrasound-images-for-breast-cancer.zip -d /content/Dataset_BUSI_with_GT/
    ```

## 6. Usage

### 6.1. Running the Main Pipeline
The `main_pipeline.py` script orchestrates the entire process from data preparation to model training and evaluation.

```bash
python main_pipeline.py
```

### 6.2. Exploring Specific Modules
Individual Python files like `yolo_pipeline.py`, `mask_extraction.py`, `feature_extraction.py`, `shape_features.py`, and `echo_scan_ensemble_model.py` can be run or imported independently for specific tasks.

### 6.3. Jupyter Notebooks / Colab
The project is designed to be highly interactive within Jupyter or Google Colab notebooks, allowing for step-by-step execution, visualization, and experimentation.

## 7. Model Performance

| Model                 | Type            | Accuracy (%) | Std Dev (%) | Parameters |
| :-------------------- | :-------------- | :----------- | :---------- | :--------- |
| Random Forest         | Classical ML    | 66.67        | ±1.07       | N/A        |
| SVM                   | Classical ML    | 60.77        | ±1.79       | N/A        |
| XGBoost (Tuned)       | Classical ML    | 67.40        | ±1.48       | N/A        |
| YOLO11 Nano           | Deep Learning   | 85.00        | —           | 2.6M       |
| YOLO11 Small          | Deep Learning   | 89.10        | —           | 6.9M       |
| Vision Transformer    | Deep Learning   | 92.31        | —           | 86M        |

**Key Insight:** Deep learning models, particularly the Vision Transformer and YOLO variants, significantly outperform classical machine learning approaches in breast ultrasound tumor classification, demonstrating their superior ability to extract and leverage complex image features.

## 8. Future Work

*   **Advanced Hyperparameter Tuning & NAS:** Utilize AutoML and Neural Architecture Search to optimize model configurations and architectures.
*   **Sophisticated Data Augmentation:** Implement domain-specific augmentations and GANs to improve generalization and address data imbalance.
*   **Explore Other Advanced Architectures:** Investigate hybrid CNN-Transformer models and self-supervised learning.
*   **Robust Ensemble Methods:** Develop and rigorously evaluate advanced ensemble strategies combining diverse models.
*   **Enhanced Interpretability (XAI):** Integrate and refine techniques like Grad-CAM and LIME for clinical trust and error analysis.
*   **Real-world Clinical Validation:** Conduct prospective studies with medical professionals.
*   **Multi-Modal Data Integration:** Incorporate clinical history and other imaging modalities.

## 9. Contributing

We welcome contributions to the EchoScan-AI project! If you're interested in improving the models, adding new features, or enhancing documentation, please refer to our `CONTRIBUTING.md` (to be created) for guidelines.

## 10. License

This project is licensed under the [MIT License](LICENSE.md) - see the `LICENSE.md` file for details.

## 11. Contact

For any inquiries or collaborations, please open an issue in this repository.
