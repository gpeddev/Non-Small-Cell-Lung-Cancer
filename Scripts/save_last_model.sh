#!/bin/bash
cd ..
NOW=$( date '+%F_%H:%M:%S' )
mkdir "./StoreResults/$NOW"

mv "./Output/Models" "./StoreResults/$NOW/"
mkdir "./Output/Models"


mv "./Output/Logs" "./StoreResults/$NOW/"
mkdir "./Output/Logs"

cat ./Models/VAE_1/VAE_1_parameters.py >> "./Output/$NOW/parameters"

mv "./Output/Images" "./StoreResults/$NOW/"

mkdir "./Output/Images"