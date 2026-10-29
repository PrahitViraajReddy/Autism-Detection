# Autism Image Classification Using Deep Learning
### An Experimental Computer Vision Research Project

An experimental deep learning project exploring image classification with **PyTorch, transfer learning, and computer vision**. The project investigates whether a pretrained convolutional neural network can distinguish between two categories of facial images in a research dataset.

> **⚠️ Important medical disclaimer**
>
> This project is for educational and experimental research purposes only. Facial-image predictions are **not a validated method for diagnosing autism** and must not be used for screening, medical decisions, or conclusions about an individual's neurodevelopment. Autism assessment requires appropriate professional evaluation.

<p align="center">
  <a href="https://huggingface.co/spaces/Mpvr/autism-detection">
    <img src="https://img.shields.io/badge/Live%20Demo-Try%20the%20App-orange?style=for-the-badge&logo=huggingface" alt="Try the live demo">
  </a>
</p>

**Live Demo:** [Autism Image Classification on Hugging Face Spaces](https://huggingface.co/spaces/Mpvr/autism-detection)

**Repository:** [Autism-Detection on GitHub](https://github.com/PrahitViraajReddy/Autism-Detection)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Technical Approach](#technical-approach)
- [Model Architecture](#model-architecture)
- [Evaluation Strategy](#evaluation-strategy)
- [Interactive Application](#interactive-application)
- [Technology Stack](#technology-stack)
- [Running Locally](#running-locally)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## Overview

This project explores a computer vision pipeline for binary image classification using deep learning.

The experimental workflow covers image preprocessing, data augmentation, transfer learning, model training, and evaluation. A pretrained **DenseNet-169** architecture is adapted for a two-class classification task involving facial images labelled *Autistic* and *Non-Autistic* in the source dataset.

An interactive application provides a way to explore image predictions and model-related information.

The primary objective is to demonstrate practical deep learning techniques and investigate the limitations of image-based classification—not to establish a clinically meaningful relationship between facial appearance and autism.

## Problem Statement

Deep learning makes it possible to learn visual patterns from labelled images. However, a model's ability to classify images in a dataset does not establish that its predictions are reliable, generalisable, or clinically meaningful.

This project explores the implementation of an image-classification pipeline while considering challenges such as:

- Image quality and variation.
- Preprocessing and augmentation choices.
- Training and validation behaviour.
- Model evaluation and error analysis.
- Potential dataset bias and limited generalisation.
- The difference between experimental classification performance and real-world validity.

The project treats these challenges as important research considerations rather than assuming that a successful classification result establishes diagnostic capability.

## Project Objectives

1. Build an image preprocessing pipeline suitable for deep learning.
2. Apply data augmentation to explore variation in training images.
3. Fine-tune a pretrained DenseNet-169 model for binary classification.
4. Monitor training and validation performance.
5. Evaluate predictions using classification metrics and visualisations.
6. Provide an interactive interface for exploring model outputs.
7. Document limitations and responsible-use considerations.

## Technical Approach

The project follows a typical transfer-learning workflow.

### 1. Dataset

The project uses the **Autistic Children Facial Image Dataset** obtained through Kaggle.

The images are associated with two classification labels. Image paths and labels are organised for use in the training pipeline.

Dataset quality, demographic representation, possible data leakage, and the suitability of the labels remain important considerations when interpreting experimental results.

### 2. Image Preprocessing

The image pipeline uses transformations intended to prepare images for model training and evaluation.

These include:

- Resizing and cropping.
- Random horizontal flipping.
- Rotation.
- Brightness, contrast, saturation, and hue adjustments where configured.
- Conversion to tensors.
- ImageNet-based normalization for transfer learning.

Augmentation is used to expose the model to variations in training images. It does not, by itself, resolve dataset bias or guarantee generalisation.

### 3. Model Training

The primary documented architecture is **DenseNet-169**, initialized with pretrained weights.

The final classification layer is adapted for two output classes.

The training workflow includes:

- Cross-entropy loss.
- Adam optimization.
- Weight decay.
- Learning-rate scheduling.
- Validation monitoring.
- Saving the best-performing checkpoint according to the configured validation criterion.

The purpose of transfer learning is to reuse visual representations learned during pretraining and adapt them to the experimental task.

### 4. Evaluation

Model evaluation can include:

- Test-set accuracy.
- Precision, recall, and F1-score.
- Classification reports.
- Confusion matrices.
- Training and validation loss curves.
- Training and validation accuracy curves.

Metrics should be reported only when they have been generated from the actual implementation and evaluated on an appropriately separated test set. No performance figure is asserted here without a verified result.

---

## Model Architecture

### DenseNet-169

DenseNet-169 is the primary architecture described for this project.

Its pretrained visual representations are adapted to produce predictions for two classes.

The broader project explores model comparison as a way to understand how different architectures behave on the same task. Architectures such as RegNet, MobileNet, Vision Transformer, and AlexNet should be considered available options only where their implementations and corresponding evaluation results are present in the repository or application.

**Important:** A model's confidence score is not the same as the probability that a person is autistic. It reflects the model's output for the configured classification task and does not establish clinical validity.

## Evaluation Strategy

A meaningful evaluation should examine more than a single accuracy value.

| Evaluation component | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified examples. |
| Precision | Measures the proportion of predicted positives that are correct within the evaluated dataset. |
| Recall | Measures the proportion of actual positives identified within the evaluated dataset. |
| F1-score | Summarises precision and recall using their harmonic mean. |
| Confusion matrix | Shows the distribution of correct and incorrect predictions by class. |
| Training and validation curves | Help investigate learning behaviour and possible overfitting. |

These metrics describe performance on the evaluated data. They do not establish that the model can diagnose autism or generalise to different populations and real-world settings.

## Interactive Application

The project includes an interactive research interface intended to make model outputs easier to explore.

Depending on the features implemented in the deployed version, the interface may provide:

- Image upload and classification.
- Predicted class and model confidence.
- Class-probability visualisation.
- Model comparison and experiment visualisations.
- Export of prediction information in JSON format.
- A separate research-assistant component using Qwen3-1.7B, where configured.

The research assistant and the image classifier serve different purposes. An assistant-generated explanation does not independently validate a classification result.

**Try the application:** [Open the Hugging Face Space](https://huggingface.co/spaces/Mpvr/autism-detection)

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| PyTorch | Deep learning and model training |
| Torchvision | Image transformations and pretrained vision models |
| Pandas | Tabular data and image-label organisation |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| Scikit-learn | Classification evaluation metrics |
| Matplotlib | Evaluation visualisations |
| Streamlit | Interactive application interface |
| Hugging Face Spaces | Hosting the live demonstration |
| Qwen3-1.7B | Separate research-assistant component, where configured |

## Running Locally

The following are the general setup steps. The exact launch command and dependencies should match the files currently included in the repository.

### Prerequisites

- Python installed.
- Git installed.
- The project's dependency file, if provided.
- Access to the dataset and any model checkpoints required by the implementation.

### 1. Clone the repository

```bash
git clone https://github.com/PrahitViraajReddy/Autism-Detection.git
cd Autism-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project's actual implementation.

### 4. Configure the dataset and model files

Place the dataset and any required checkpoints in the locations expected by the code. Follow the paths and configuration options defined in the repository.

Do not assume that the dataset or trained weights are included in the repository unless the files are present.

### 5. Launch the application

For a Streamlit application whose entry point is `app.py`, use:

```bash
streamlit run app.py
```

If the deployed application uses a different entry point, use the launch command specified by that implementation.

---

## Limitations and Responsible Use

This project has important limitations.

### Experimental classification is not diagnosis

Autism is a complex neurodevelopmental condition. A model that classifies facial images cannot establish whether an individual is autistic.

### Dataset limitations

The results depend on the dataset, its labels, image quality, class composition, and collection process. Potential sampling bias and demographic imbalance may affect the learned patterns.

### Generalisation

Performance on a particular dataset does not guarantee performance on new images, different populations, or images collected under different conditions.

### Evaluation requirements

Reliable experimental conclusions require careful separation of training, validation, and test data; checks for data leakage; transparent metric reporting; and further investigation of model errors and bias.

### Confidence is not clinical certainty

A high-confidence prediction does not demonstrate that a prediction is correct or medically meaningful.

**This project must not be used for autism diagnosis, medical screening, treatment decisions, or labelling individuals.** It is an educational demonstration of deep learning and computer vision techniques.

## Key Learnings

This project provides practical experience with:

- Building image preprocessing and augmentation pipelines.
- Applying transfer learning with pretrained neural networks.
- Configuring model training and validation.
- Monitoring experiments and saving checkpoints.
- Evaluating classification models using multiple metrics.
- Connecting a deep learning workflow to an interactive application.
- Communicating technical limitations and responsible-use considerations.

## Future Improvements

Potential improvements include:

- More rigorous checks for data leakage.
- Reproducible training and evaluation configurations.
- Detailed error analysis and per-class metrics.
- Better documentation of dataset provenance and limitations.
- Evaluation across independent datasets, where appropriate data and permissions are available.
- Improved reproducibility through documented dependencies and model checkpoints.
- Clearer separation between classification outputs and research-assistant functionality.

These are potential directions for further work, not claims that the improvements have already been implemented.

---

## Project Links

- **Live Demo:** https://huggingface.co/spaces/Mpvr/autism-detection
- **GitHub Repository:** https://github.com/PrahitViraajReddy/Autism-Detection

---

*An experimental computer vision project focused on deep learning, transfer learning, model evaluation, and responsible communication of research limitations.*
