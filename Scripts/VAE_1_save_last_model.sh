#!/bin/bash
cd ..
NOW=$( date '+%F_%H:%M:%S' )
mkdir "./StoreResults/VAE_1/$NOW"

mv "./Output/VAE_1/Models" "./StoreResults/VAE_1/$NOW/"
mkdir "./Output/VAE_1/Models"

mv "./Output/VAE_1/Logs" "./StoreResults/VAE_1/$NOW/"
mkdir "./Output/VAE_1/Logs"

mv "./Output/VAE_1/Images" "./StoreResults/VAE_1/$NOW/"
mkdir "./Output/VAE_1/Images"
mkdir "./Output/VAE_1/Images/0"
mkdir "./Output/VAE_1/Images/1"
mkdir "./Output/VAE_1/Images/2"
mkdir "./Output/VAE_1/Images/3"
mkdir "./Output/VAE_1/Images/4"


mv "./Output/VAE_1/DatasetSplits" "./StoreResults/VAE_1/$NOW/"
mkdir "./Output/VAE_1/DatasetSplits"

mv "./Output/VAE_1/hyperparameters.txt" "./StoreResults/VAE_1/$NOW/"
