# Unsupervised Feature Extraction for Assessing Recurrence of Lung Cancer (NSCLC)

> MSc Data Analytics thesis project (University of Glasgow)  
> Author: **Georgios Pediaditis**

This repository contains code to predict **recurrence** in Non-Small-Cell Lung Cancer (NSCLC) from **CT imaging** (with optional clinical/genetic covariates). The core idea is to use **Variational Autoencoders (VAEs)** as *unsupervised feature extractors* and train a downstream **SVM** classifier. Two complementary CT strategies are implemented:

1) **Tumor-only** (expert mask applied)  
2) **Window/Bounding-box** around the tumor (less mask-dependent)

The pipeline includes image preprocessing, VAE training & hyper-parameter search, feature pooling across slices, L1-logistic feature selection, class balancing with SMOTE, and SVM classification with cross-validation.

> ⚠️ **Research-use only.** This code is for academic research and is **not** a medical device. Do not use it for clinical decision-making.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Environment & Requirements](#environment--requirements)
- [Data](#data)
- [Workflow](#workflow)
- [Scripts Cheat-Sheet](#scripts-cheat-sheet)
- [Results (High Level)](#results-high-level)
- [Reproducibility Tips](#reproducibility-tips)
- [Citing This Work](#citing-this-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Quick Start

```bash
# 1) Create and activate an environment
conda create -n nsclc python=3.9 -y
conda activate nsclc

# 2) Install core deps (adjust versions to your setup)
pip install tensorflow==2.* numpy pandas scikit-learn imbalanced-learn             SimpleITK pydicom opencv-python matplotlib tqdm

# 3) Prepare your data directories (see "Data" below) and set paths in the scripts

# 4) (Optional) Check GPU visibility
python TestGPU.py

# 5) Preprocess CTs (windowing, masking / window boxes, crops)
python Preprocessing.py    # edit path variables or args inside as needed

# 6) Train VAEs (choose one or more models/variants)
python VAE_1_tunning_model_1.py   # tumor-only, model 1
python VAE_1_tunning_model_2a.py  # tumor-only, model 2
python VAE_2_tunning_model_1.py   # window/box, model 1
python VAE_2_tunning_model_2.py   # window/box, model 2
# (also available) python VAE_2_tunning.py

# 7) Extract & pool features across slices, then select features + classify
python FeatureSelection_Classification.py          # L1-logistic feature selection
python Classification_LassoSmoteSVM.py            # SMOTE + SVM training/eval

# 8) (Optional) Visualize reconstructions/latent behavior
python Visualization_VAE1.py
python Visualization_VAE2.py
```

> The scripts are designed to be edited for your local paths/config. If you prefer CLI flags, you can easily wrap path variables with `argparse`.

---

## Project Structure

Top-level layout (abridged):

```
.
├─ ClinicalDataAnalysis/           # notebooks/scripts exploring clinical & gene data
├─ Models/                         # saved model weights/checkpoints (you create these)
├─ Pooling/                        # utilities to pool slice-wise embeddings to patient-level
├─ ProcessingCTs/                  # helpers for CT windowing, masking, cropping
├─ Scripts/                        # extra utilities / convenience scripts
├─ SupportCode/                    # shared functions (metrics, loaders, transforms, etc.)
├─ CancerRecurrence.py             # end-to-end orchestration (optional usage)
├─ Preprocessing.py                # CT preprocessing + mask/box pipelines
├─ VAE_1_tunning_model_1.py        # tumor-only VAE (model 1) + tuning
├─ VAE_1_tunning_model_2a.py       # tumor-only VAE (model 2) + tuning
├─ VAE_2_tunning.py                # window/box VAE generic + tuning
├─ VAE_2_tunning_model_1.py        # window/box VAE (model 1) + tuning
├─ VAE_2_tunning_model_2.py        # window/box VAE (model 2) + tuning
├─ FeatureSelection_Classification.py
├─ Classification_LassoSmoteSVM.py
├─ PCA_FAMD_Classification.py      # optional dimensionality-reduction experiments
├─ SSIM calculator.py              # SSIM utilities for VAE comparison
├─ Visualization_VAE1.py           # plots / reconstructions for VAE1
├─ Visualization_VAE2.py           # plots / reconstructions for VAE2
└─ TestGPU.py                      # quick sanity check for TF GPU
```

> Exact responsibilities of each script are annotated in-file. Most training hyper-parameters and paths are set at the top of each script for clarity.

---

## Environment & Requirements

- **Python**: 3.8–3.10 recommended  
- **Deep learning**: TensorFlow 2.x (or adapt to PyTorch if you wish)  
- **Core packages**:
  - `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn` (SMOTE)
  - `SimpleITK` and/or `pydicom` for medical image IO
  - `opencv-python` (image ops), `matplotlib` (plots), `tqdm` (progress)

> GPU highly recommended for VAE training. Use `TestGPU.py` to verify your CUDA/CuDNN/TensorFlow stack.

---

## Data

This project targets the **Radiogenomic NSCLC** cohort (CTs + masks, clinical & genetic metadata).  
You will need:

- Raw **CT scans** (DICOM or other supported format)
- **Tumor masks** (for the tumor-only pipeline)  
- Optional **clinical/genetic CSVs** (for the multimodal variant)

Suggested structure:

```
data/
  ct/           # per-patient folders with CT series
  masks/        # per-patient tumor masks (if available)
  clinical/     # CSVs with clinical/genetic covariates (optional)
```

Update the path variables inside `Preprocessing.py` and the training scripts to match your setup.

---

## Workflow

1) **Preprocessing**
   - CT **windowing** to lung settings, grayscale conversion, type casting
   - **Tumor-only** path: apply expert masks → crop around tumor (fixed window size)  
   - **Window/box** path: expand masks to a box around the tumor center → apply to CT
   - Save per-slice arrays ready for model consumption

2) **Modeling: VAEs (unsupervised)**
   - Several lightweight 2D VAE architectures (two per strategy) with manual grid search on:
     - filters / latent size
     - learning rate
     - KL weight  
   - Compare models across folds using **reconstruction loss** and **SSIM** (for cross-KL comparability)

3) **Feature extraction & pooling**
   - Pass **all slices** of each patient through the **encoder**  
   - **Pool** slice-level embeddings to **patient-level** features (e.g., max & mean pooling)

4) **Feature selection & class balancing**
   - **L1-logistic regression** to zero-out unhelpful features
   - **SMOTE** to rebalance the minority recurrence class on the training fold

5) **Classification & evaluation**
   - Train an **SVM** on the selected features
   - 5-fold CV **by patient** (not slice)  
   - Report **AUC**, **sensitivity**, **specificity**, confusion matrices

---

## Scripts Cheat-Sheet

- **Preprocessing**
  - [`Preprocessing.py`](./Preprocessing.py) – main CT pipeline (windowing, crops/boxes, saving tensors)

- **VAE training & selection**
  - [`VAE_1_tunning_model_1.py`](./VAE_1_tunning_model_1.py), [`VAE_1_tunning_model_2a.py`](./VAE_1_tunning_model_2a.py) – tumor-only VAEs  
  - [`VAE_2_tunning_model_1.py`](./VAE_2_tunning_model_1.py), [`VAE_2_tunning_model_2.py`](./VAE_2_tunning_model_2.py) – window/box VAEs  
  - [`VAE_2_tunning.py`](./VAE_2_tunning.py) – generic window/box VAE tuning
  - `SSIM calculator.py` – SSIM utilities & comparisons across KL weights
  - [`Visualization_VAE1.py`](./Visualization_VAE1.py), [`Visualization_VAE2.py`](./Visualization_VAE2.py) – reconstructions/latents

- **Downstream ML**
  - [`FeatureSelection_Classification.py`](./FeatureSelection_Classification.py) – L1-logistic feature selection  
  - [`Classification_LassoSmoteSVM.py`](./Classification_LassoSmoteSVM.py) – SMOTE + SVM training & metrics  
  - [`PCA_FAMD_Classification.py`](./PCA_FAMD_Classification.py) – optional DR baselines (PCA, FAMD)

- **Misc**
  - [`CancerRecurrence.py`](./CancerRecurrence.py) – optional end-to-end runner  
  - [`TestGPU.py`](./TestGPU.py) – TensorFlow GPU visibility check

---

## Results (High Level)

- **Both** pipelines show predictive signal from **VAE features** alone.  
- The **window/bounding-box** strategy tends to perform **better overall** than strict tumor-only masking in the final SVM stage (helpful context retained).  
- Adding raw clinical/genetic covariates **without further dimensionality reduction** can **hurt** performance on small samples (overfitting risk).  
- See the figures and per-fold reports produced by the visualization & classification scripts for full details once you run them locally.

> Tip: start with the window/box VAE variant to reproduce the strongest baseline, then add clinical data only after aggressive feature selection or DR.

---

## Reproducibility Tips

- Keep **fold splits by patient**, not by slice.  
- Fix random seeds across NumPy/TensorFlow/scikit-learn where possible.  
- Save **model checkpoints** and **feature CSVs** per fold for traceability.  
- Log **hyper-parameters** (filters, latent size, KL weight, LR) and **SSIM** per model selection step.  
- Record **class distributions** before/after SMOTE for each fold.

---

## Citing This Work

If you use this code or pipeline, please cite:

```
@thesis{Pediaditis2025NSCLC,
  author    = {Georgios Pediaditis},
  title     = {Unsupervised Feature Extraction for Assessing Recurrence of Lung Cancer},
  school    = {University of Glasgow, School of Mathematics and Statistics},
  year      = {2025},
  note      = {MSc Data Analytics Thesis},
}
```

Key references (methods & data):
- D. P. Kingma, M. Welling — Auto-Encoding Variational Bayes (2014)  
- S. Bakr et al. — A radiogenomic dataset of non-small cell lung cancer (Sci Data, 2018)

---

## License

No license specified yet. If you intend others to use or extend this work, add a `LICENSE` file at the repo root (e.g., MIT, Apache-2.0).

---

## Acknowledgements

- The Cancer Imaging Archive and contributors of the Radiogenomic NSCLC cohort  
- Supervisors/colleagues who supported this MSc research  
- Open-source libraries used throughout (TensorFlow, scikit-learn, SimpleITK, etc.)
