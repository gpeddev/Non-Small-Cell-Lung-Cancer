#!/bin/bash
cd ..
NOW=$( date '+%F_%H:%M:%S' )
mkdir "./Results/$NOW"

mv "./SavedModels" "./Results/$NOW/"
mkdir "./SavedModels"


mv "./Logs/" "./Results/$NOW/"
mkdir "./Logs"

cat ./Models/VAE_1/VAE_1_parameters.py >> "./Results/$NOW/parameters"



mv "./Images" "./Results/$NOW/"

mkdir "./Images"