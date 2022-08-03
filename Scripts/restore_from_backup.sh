#!/bin/bash

cd ..
mkdir ./Data/00_Images
mkdir ./Data/01_Masks
mkdir ./Data/02_WindowCT_VAE1_results
mkdir ./Data/03_GrayscaleCT_VAE1_results
mkdir ./Data/04_MaskCT_VAE1_results
mkdir ./Data/05_CropTumor_VAE1_results
mkdir ./Data/06_NewMaskWindow_VAE2_results
mkdir ./Data/07_MaskedCT_VAE2_results
mkdir ./Data/08_CroppedWindow_VAE2_results
mkdir ./Data/09_TrainingSet_VAE1
mkdir ./Data/10_TrainingSet_VAE2

cp ./Data/Backup_Data/*.nii ./Data/00_Images/
mv ./Data/00_Images/*_roi.nii ./Data/01_Masks/
