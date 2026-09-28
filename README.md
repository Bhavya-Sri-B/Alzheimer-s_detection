
This project presents a deep learning-based framework for early detection of Alzheimer’s disease using structural brain MRI scans.
A customized 3D Convolutional Neural Network (3D CNN) is used to learn volumetric features from MRI scans and classify subjects into Alzheimer’s Disease (AD) and Cognitively Normal (CN) categories.
The model achieved 96.4% accuracy, 95.1% sensitivity, 97.2% specificity, 95.5% F1-score, and 0.97 AUC on the reported test dataset.

## Objectives
* Detect Alzheimer’s disease from structural brain MRI scans.
* Apply standardized MRI preprocessing techniques.
* Extract volumetric features using a customized 3D CNN.
* Improve model robustness using data augmentation and regularization.
* Evaluate performance using standard classification metrics.

## Methodology

'''text
MRI Scans
    ↓
Preprocessing
    ↓
Data Augmentation
    ↓
Customized 3D CNN
    ↓
Feature Extraction
    ↓
Global Average Pooling
    ↓
Softmax Classification
    ↓
AD / CN Prediction
```

## MRI Preprocessing

* Reorientation and resampling
* Intensity normalization
* Skull stripping
* Spatial alignment
* Cropping / padding

## Data Augmentation

* Random rotations
* Spatial translations
* Anatomically valid flipping
* Intensity perturbations

## 3D CNN Architecture

The customized CNN uses:
* 3D Convolutional layers
* Batch Normalization
* ReLU activation
* 3D Max Pooling
* Global Average Pooling
* Fully Connected Classification Layer
* Softmax output

The model learns hierarchical spatial features from volumetric MRI data for subject-level classification.

## Training

* Optimizer: Adam
* Loss Function: Categorical Cross-Entropy
* Regularization: Dropout & Weight Decay
* Overfitting Control: Early Stopping
* Learning Rate: Validation-based scheduling

##  Results

| Metric      |     Score |
| ----------- | --------: |
| Accuracy    | **96.4%** |
| Sensitivity | **95.1%** |
| Specificity | **97.2%** |
| Precision   | **96.0%** |
| F1-Score    | **95.5%** |
| AUC         |  **0.97** |

## Technologies & Concepts
* Python
* Deep Learning
* 3D Convolutional Neural Networks
* Medical Image Processing
* Brain MRI Analysis
* Data Augmentation
* Classification & Model Evaluation

## Future Scope
* Validate the model on larger and more diverse datasets.
* Integrate multimodal medical imaging and clinical biomarkers.
* Apply domain adaptation for different clinical environments.
* Incorporate explainable AI techniques such as saliency mapping and activation visualization.
