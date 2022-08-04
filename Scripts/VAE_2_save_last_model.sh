#!/bin/bash
cd ..
NOW=$( date '+%F_%H:%M:%S' )
mkdir "./StoreResults/VAE_2/$NOW"

mv "./Output/VAE_2/Models" "./StoreResults/VAE_2/$NOW/"
mkdir "./Output/VAE_2/Models"

mv "./Output/VAE_2/Logs" "./StoreResults/VAE_2/$NOW/"
mkdir "./Output/VAE_2/Logs"

mv "./Output/VAE_2/Images" "./StoreResults/VAE_2/$NOW/"
mkdir "./Output/VAE_2/Images"
mkdir "./Output/VAE_2/Images/0"
mkdir "./Output/VAE_2/Images/1"
mkdir "./Output/VAE_2/Images/2"
mkdir "./Output/VAE_2/Images/3"
mkdir "./Output/VAE_2/Images/4"


mv "./Output/VAE_2/DatasetSplits" "./StoreResults/VAE_2/$NOW/"
mkdir "./Output/VAE_2/DatasetSplits"

mv "./Output/VAE_2/hyperparameters.txt" "./StoreResults/VAE_2/$NOW/"

mv "./Output/VAE_2/vae_2.png" "./StoreResults/$NOW/"