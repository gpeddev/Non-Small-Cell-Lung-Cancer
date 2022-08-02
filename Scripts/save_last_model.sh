#!/bin/bash
cd ..
NOW=$( date '+%F_%H:%M:%S' )
mkdir "./StoreResults/$NOW"

mv "./Output/Models" "./StoreResults/$NOW/"
mkdir "./Output/Models"

mv "./Output/Logs" "./StoreResults/$NOW/"
mkdir "./Output/Logs"

mv "./Output/Images" "./StoreResults/$NOW/"
mkdir "./Output/Images"
mkdir "./Output/Images/0"
mkdir "./Output/Images/1"
mkdir "./Output/Images/2"
mkdir "./Output/Images/3"
mkdir "./Output/Images/4"


mv "./Output/DatasetSplits" "./StoreResults/$NOW/"
mkdir "./Output/DatasetSplits"

mv "./Output/hyperparameters.txt" "./StoreResults/$NOW/"
