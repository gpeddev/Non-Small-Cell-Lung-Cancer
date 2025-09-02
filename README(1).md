# Unsupervised Feature Extraction for Assessing Recurrence of Lung Cancer

This repository contains the code developed as part of my MSc Data Analytics thesis at the University of Glasgow.  
The project explores **unsupervised feature extraction from CT scans** of lung cancer patients to predict **recurrence of non-small-cell lung cancer (NSCLC)** after surgery.  

By combining **medical imaging, clinical data, and genetic data**, the study investigates how **variational autoencoders (VAEs)** can be used to extract predictive features that improve prognosis models.

---

## 🚀 Project Overview

Lung cancer remains one of the leading causes of death worldwide. Even after surgery, **30–55% of NSCLC patients relapse**, making recurrence prediction a critical clinical challenge.  
This project applies **deep learning techniques** to:

- Extract features from CT scans using **Variational Autoencoders (VAE)**.  
- Compare two segmentation approaches:  
  - **Pixel-level segmentation** via expert tumor masks.  
  - **Bounding-box segmentation** around the tumor region.  
- Fuse imaging features with **clinical and genetic variables**.  
- Train **Support Vector Machines (SVMs)** for recurrence classification.  

---

## 🧑‍💻 Methodology

1. **Data Source**  
   - Sourced publicly from the **Cancer Imaging Archive (TCIA)**.  
   - Includes CT scans, tumor annotations, clinical metadata, and genetic markers.  
   - Final dataset comprises **142 patients**.  

2. **Preprocessing**  
   - CT windowing and normalization.  
   - Tumor cropping (both masks and bounding boxes).  
   - Data augmentation: flipping, rotation, etc.

3. **Feature Extraction & Modeling**  
   - Train custom **2D VAE models** using 5-fold cross-validation.  
   - Aggregate latent features across slices using **max/average pooling**.  
   - Use **Lasso** for feature selection; apply **SMOTE** to address class imbalance.  
   - Classify recurrence outcomes using SVMs; evaluate using **AUC, sensitivity, and specificity**.

---

## 📊 Results

- **Bounding-box VAEs** consistently outperformed pixel-level models.  
- Best achieved results:  
  - **AUC ≈ 0.81** — imaging features via bounding box.  
  - **AUC ≈ 0.72** — imaging features via pixel masks.  
- Adding clinical/genetic data had mixed effects—possibly due to high dimensionality versus sample size.  

**Takeaway**: Variational autoencoder-derived features, especially with bounding-box segmentation, are promising predictors of NSCLC recurrence.

---

## 🛠️ Tech Stack

- **Python 3.8+**  
- **TensorFlow / Keras** – for VAE implementations  
- **scikit-learn** – for SVM classification and Lasso feature selection  
- **imblearn** – for SMOTE resampling  
- **NumPy / Pandas** – data operations  
- **Matplotlib / Seaborn** – result visualization  

---

## 📂 Repository Structure

```
├── .idea/                           # IDE configuration files
├── ClinicalDataAnalysis/            # Scripts and tools for clinical/genetic data handling
├── Models/                          # VAE model definitions and training utilities
├── Pooling/                         # Code for pooling latent features across CT slices
├── ProcessingCTs/                   # CT image preprocessing scripts
├── Scripts/                         # Utility scripts (e.g., training, evaluation, plotting)
├── SupportCode/                     # Helper modules and shared utilities
├── CancerRecurrence.py              # Pipeline to combine features and assess recurrence
├── Classification_LassoSmoteSVM.py  # Lasso + SMOTE + SVM classification script
├── FeatureSelection_Classification.py # Feature selection and classification experiments
├── PCA_FAMD_Classification.py       # Dimensionality reduction with PCA/FAMD for classification
├── Preprocessing.py                 # Main preprocessing pipeline
├── SSIM calculator.py               # Structural similarity calculations (for VAE evaluation)
├── TestGPU.py                       # GPU performance testing script
├── VAE_1_tunning_model_1.py         # First VAE architecture tuning script
├── VAE_1_tunning_model_2a.py        # Alternative VAE tuning iteration
├── VAE_2_tunning.py                 # Second VAE architecture tuning
├── VAE_2_tunning_model_1.py         # First tuning version of second VAE design
├── VAE_2_tunning_model_2.py         # Second tuning version of second VAE design
├── Visualization_VAE1.py            # Latent space and reconstruction visualization (VAE1)
└── Visualization_VAE2.py            # Visualization for VAE2 outputs
├── .gitignore                       # File patterns to ignore
```
