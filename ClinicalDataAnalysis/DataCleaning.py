import pandas as pd
import numpy as np
import os
pd.options.mode.chained_assignment = None

# load dataframe
data = pd.read_csv("./ClinicalData/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
data.dtypes

# Remove patients we dont need
patient_namelist = os.listdir("./Data/00_Images")
patient_namelist = [patient.rsplit(".")[0] for patient in patient_namelist]     # removes .nii
data = data[data['Case ID'].isin(patient_namelist)]

# Exploratory Analysis
data.head()
data.count()
# drop irrelevant variables
# Date of Last Known Alive
# Survival Status
# Date of Death
# Time to Death (days)
# CT Date
# PET Date
data.drop("Patient affiliation",inplace=True,axis=1)
data.drop("Date of Last Known Alive",inplace=True,axis=1)
data.drop("Survival Status",inplace=True,axis=1)
data.drop("Date of Death",inplace=True,axis=1)
data.drop("Time to Death (days)",inplace=True,axis=1)
data.drop("CT Date",inplace=True,axis=1)
data.drop("PET Date",inplace=True,axis=1)

# Remaining variables
data.dtypes


# Drop Quit Smoking Year
data.drop("Quit Smoking Year",inplace=True,axis=1)

# pack years
# replace non smokers nan with 0
data["Pack Years"]=data["Pack Years"].replace(np.nan,0)

# replace missing values by mean
data.loc[:,'Pack Years'] = pd.to_numeric(data['Pack Years'], errors='coerce')
data.describe()
data["Pack Years"]=data["Pack Years"].replace(np.nan,35.175182)

# Lymphovascular invasion
data.drop("Lymphovascular invasion",inplace=True,axis=1)

# ALK translocation status
data.drop("ALK translocation status",inplace=True,axis=1)

# EGFR mutation status
data.drop("EGFR mutation status",inplace=True,axis=1)

# KRAS mutation status
data.drop("KRAS mutation status",inplace=True,axis=1)

data.dtypes
data.drop("Tumor Location (choice=Unknown)",inplace=True,axis=1)

# weight
data.loc[:,'Weight (lbs)'] = pd.to_numeric(data['Weight (lbs)'], errors='coerce')

data["Weight (lbs)"].describe()
data['Weight (lbs)'] = data['Weight (lbs)'].replace(np.nan, 172.154995)     # replace with mean

data.dtypes


# %GG
data["%GG"].unique()

# # Ordinal data replaced by 0,1,2,3,4,5
# data.loc[:,"%GG"] = data["%GG"].replace("0%",0)
# data.loc[:,"%GG"] = data["%GG"].replace(">0 - 25%",1)
# data.loc[:,"%GG"] = data["%GG"].replace("50 - 75%",2)
# data.loc[:,"%GG"] = data["%GG"].replace("25 - 50%",3)
# data.loc[:,"%GG"] = data["%GG"].replace("75 - < 100%",4)
# data.loc[:,"%GG"] = data["%GG"].replace("100%",5)
data["%GG"].unique()
data.dtypes

# Gender
data.loc[:,"Gender"] = pd.Categorical(data["Gender"])

# Ethnicity
data["Ethnicity"].unique()
data.loc[:,"Ethnicity"] = pd.Categorical(data["Ethnicity"])

# Smoking status
data["Smoking status"].unique()
data.loc[:,"Smoking status"] = pd.Categorical(data["Smoking status"])

# Tumor Location (choice=RUL)
data["Tumor Location (choice=RUL)"].unique()
data.loc[:,"Tumor Location (choice=RUL)"] = pd.Categorical(data["Tumor Location (choice=RUL)"])

# Tumor Location (choice=RML)
data["Tumor Location (choice=RML)"].unique()
data.loc[:,"Tumor Location (choice=RML)"] = pd.Categorical(data["Tumor Location (choice=RML)"])

# Tumor Location (choice=RUL)
data["Tumor Location (choice=RUL)"].unique()
data.loc[:,"Tumor Location (choice=RLL)"] = pd.Categorical(data["Tumor Location (choice=RLL)"])

# Tumor Location (choice=LUL)
data["Tumor Location (choice=LUL)"].unique()
data.loc[:,"Tumor Location (choice=LUL)"] = pd.Categorical(data["Tumor Location (choice=LUL)"])

# Tumor Location (choice=LLL)
data["Tumor Location (choice=LLL)"].unique()
data.loc[:,"Tumor Location (choice=LLL)"] = pd.Categorical(data["Tumor Location (choice=LLL)"])

# Tumor Location (choice=L Lingula)
data["Tumor Location (choice=L Lingula)"].unique()
data.loc[:,"Tumor Location (choice=L Lingula)"] = pd.Categorical(data["Tumor Location (choice=L Lingula)"])

data.dtypes

# Histology
data["Histology"].unique()
data.loc[:,"Histology"] = pd.Categorical(data["Histology"])

# Pathological T stage
data["Pathological T stage"].unique()
data.loc[:,"Pathological T stage"] = pd.Categorical(data["Pathological T stage"])

# Pathological N stage
data["Pathological N stage"].unique()
data.loc[:,"Pathological N stage"] = pd.Categorical(data["Pathological N stage"])

# Pathological M stage
data["Pathological M stage"].unique()
data.loc[:,"Pathological M stage"] = pd.Categorical(data["Pathological M stage"])

# Histopathological Grade
data["Histopathological Grade"].unique()
data.loc[:,"Histopathological Grade"] = pd.Categorical(data["Histopathological Grade"])

# Pleural invasion (elastic, visceral, or parietal)
data["Pleural invasion (elastic, visceral, or parietal)"].unique()
data.loc[:,"Pleural invasion (elastic, visceral, or parietal)"] = pd.Categorical(data["Pleural invasion (elastic, visceral, or parietal)"])

# Adjuvant Treatment
data["Adjuvant Treatment"].unique()
data.loc[:,"Adjuvant Treatment"] = pd.Categorical(data["Adjuvant Treatment"])

# Chemotherapy
data["Chemotherapy"].unique()
data.loc[:,"Chemotherapy"] = pd.Categorical(data["Chemotherapy"])

# Radiation
data["Radiation"].unique()
data.loc[:,"Radiation"] = pd.Categorical(data["Radiation"])

# Recurrence
data["Recurrence"].unique()
data.loc[:,"Recurrence"] = pd.Categorical(data["Recurrence"])

data["Date of Recurrence"].unique()
data.drop("Date of Recurrence",inplace=True,axis=1)

data.drop("Recurrence Location",inplace=True,axis=1)
data.dtypes
data.to_csv("./ClinicalDataAnalysis/data.csv",header=True,index=False)
