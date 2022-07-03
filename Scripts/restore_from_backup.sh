#!/bin/bash

cd ..
cp ./Data/Backup_Data/*.nii ./Data/00_Images/
mv ./Data/00_Images/*_roi.nii ./Data/01_Masks/
