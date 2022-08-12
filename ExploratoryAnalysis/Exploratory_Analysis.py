import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None

# load dataframe
data = pd.read_csv("./ClinicalData/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")

# Remove patients we dont need
patient_namelist = os.listdir("./Data/00_Images")
patient_namelist = [patient.rsplit(".")[0] for patient in patient_namelist]     # removes .nii
data = data[data['Case ID'].isin(patient_namelist)]

# Exploratory Analysis
data.head()

# Patient affiliation
data["Patient affiliation"].unique()

# Age at Histological Diagnosis
data["Age at Histological Diagnosis"].unique()
data["Age at Histological Diagnosis"].describe()

# Weight (lbs)
data["Weight (lbs)"].unique()

data.loc[:,'Weight (lbs)'] = pd.to_numeric(data['Weight (lbs)'], errors='coerce')

data["Weight (lbs)"].describe()
data['Weight (lbs)'] = data['Weight (lbs)'].replace(np.nan, 172.154995)     # replace with mean

# Gender
data["Gender"].unique()

#Ethnicity
data["Ethnicity"].unique()

# percentage of each group
data["Ethnicity"].value_counts()/data["Ethnicity"].count()*100

# create pie chart for ethnicity
values = data["Ethnicity"].value_counts()
labels = data["Ethnicity"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()

# Smoking status
data["Smoking status"].unique()
data["Smoking status"].value_counts()

# percentage of smoking status
data["Smoking status"].value_counts()/data["Smoking status"].count()*100


# Pie chart for smoking status
values = data["Smoking status"].value_counts()
labels = data["Smoking status"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()

# Pack Years
data["Pack Years"].unique()
data.loc[:,'Pack Years'] = pd.to_numeric(data['Pack Years'], errors='coerce')
data["Pack Years"].unique()
data["Pack Years"].describe()
data['Pack Years'] = data['Pack Years'].replace(np.nan, 41.904348)     # replace with mean
data["Pack Years"].unique()
data["Pack Years"].describe()
data.dtypes

# Quit Smoking Year
data["Quit Smoking Year"].unique()

# %GG
data["%GG"].unique()
# make pie chart
values = data["%GG"].value_counts()
labels = data["%GG"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()
# Ordinal data replaced by 0,1,2,3,4,5
data.loc[:,"%GG"] = data["%GG"].replace("0%",0)
data.loc[:,"%GG"] = data["%GG"].replace(">0 - 25%",1)
data.loc[:,"%GG"] = data["%GG"].replace("50 - 75%",2)
data.loc[:,"%GG"] = data["%GG"].replace("25 - 50%",3)
data.loc[:,"%GG"] = data["%GG"].replace("75 - < 100%",4)
data.loc[:,"%GG"] = data["%GG"].replace("100%",5)
data["%GG"].unique()
data.dtypes

# Tumor Location (choice=RUL)
data["Tumor Location (choice=RUL)"].unique()
data.loc[:,"Tumor Location (choice=RUL)"] = pd.Categorical(data["Tumor Location (choice=RUL)"])
data.dtypes

data["Tumor Location (choice=RML)"].unique()
data.loc[:,"Tumor Location (choice=RML)"] = pd.Categorical(data["Tumor Location (choice=RML)"])
data.dtypes

data["Tumor Location (choice=RLL)"].unique()
data.loc[:,"Tumor Location (choice=RLL)"] = pd.Categorical(data["Tumor Location (choice=RLL)"])
data.dtypes

data["Tumor Location (choice=LUL)"].unique()
data.loc[:,"Tumor Location (choice=LUL)"] = pd.Categorical(data["Tumor Location (choice=LUL)"])
data.dtypes

data["Tumor Location (choice=LLL)"].unique()
data.loc[:,"Tumor Location (choice=LLL)"] = pd.Categorical(data["Tumor Location (choice=LLL)"])
data.dtypes

data["Tumor Location (choice=L Lingula)"].unique()
data.loc[:,"Tumor Location (choice=L Lingula)"] = pd.Categorical(data["Tumor Location (choice=L Lingula)"])
data.dtypes

data["Tumor Location (choice=Unknown)"].unique()
data.loc[:,"Tumor Location (choice=Unknown)"] = pd.Categorical(data["Tumor Location (choice=Unknown)"])
data.dtypes
data=data.drop("Tumor Location (choice=Unknown)", axis=1)

# Histology
data["Histology"].unique()
data["Histology"].value_counts()

# percentage of Histology
data["Histology"].value_counts()/data["Histology"].count()*100

# Pie chart for Histology
values = data["Histology"].value_counts()
labels = data["Histology"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()
data.loc[:,"Histology"] = pd.Categorical(data["Histology"])


# Pathological T stage
data["Pathological T stage"].unique()
data.loc[:,"Pathological T stage"] = pd.Categorical(data["Pathological T stage"])
data.dtypes

# Pathological N stage
data["Pathological N stage"].unique()
data.loc[:,"Pathological N stage"] = pd.Categorical(data["Pathological N stage"])
data.dtypes

# Pathological M stage
data["Pathological M stage"].unique()
data.loc[:,"Pathological M stage"] = pd.Categorical(data["Pathological M stage"])
data.dtypes

# Histopathological Grade
data["Histopathological Grade"].unique()
data.loc[:,"Histopathological Grade"] = pd.Categorical(data["Histopathological Grade"])
data.dtypes

# Lymphovascular invasion
data["Lymphovascular invasion"].unique()
data.loc[:,"Lymphovascular invasion"] = pd.Categorical(data["Lymphovascular invasion"])

# Pleural invasion (elastic, visceral, or parietal)
data["Pleural invasion (elastic, visceral, or parietal)"].unique()
data.loc[:,"Pleural invasion (elastic, visceral, or parietal)"] = pd.Categorical(data["Pleural invasion (elastic, visceral, or parietal)"])

# EGFR mutation status
data["EGFR mutation status"].unique()
data["EGFR mutation status"].value_counts()
data.loc[:,"EGFR mutation status"] = pd.Categorical(data["EGFR mutation status"])


# KRAS mutation status
data["KRAS mutation status"].unique()
data["KRAS mutation status"].value_counts()
data.loc[:,"KRAS mutation status"] = pd.Categorical(data["KRAS mutation status"])

# ALK translocation status
data["ALK translocation status"].unique()
data["ALK translocation status"].value_counts()
data.loc[:,"ALK translocation status"] = pd.Categorical(data["ALK translocation status"])

# Adjuvant Treatment
data["Adjuvant Treatment"].unique()

# Plot of adjuvant treatment
values = data["Adjuvant Treatment"].value_counts()
labels = data["Adjuvant Treatment"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()
data.loc[:,"Adjuvant Treatment"] = pd.Categorical(data["Adjuvant Treatment"])
data["Adjuvant Treatment"].value_counts()

# to categorical
data.loc[:,"Adjuvant Treatment"] = pd.Categorical(data["Adjuvant Treatment"])

# Chemotherapy
data.loc[:,"Chemotherapy"] = pd.Categorical(data["Chemotherapy"])
data["Chemotherapy"].unique()
data["Chemotherapy"].value_counts()
data.dtypes

# Radiation
data["Radiation"].unique()
data["Radiation"].value_counts()
data.loc[:,"Radiation"] = pd.Categorical(data["Radiation"])

# Recurrence
data["Recurrence"].unique()
data["Recurrence"].value_counts()
data.loc[:,"Recurrence"] = pd.Categorical(data["Recurrence"])

# percentage recurrence
data["Recurrence"].value_counts()/data["Recurrence"].count()*100


# Pie chart for Recurrence
values = data["Recurrence"].value_counts()
labels = data["Recurrence"].unique().tolist()
plt.pie(values,labels=labels, radius=1)
plt.show()
data.loc[:,"Recurrence"] = pd.Categorical(data["Recurrence"])

# Recurrence Location
data["Recurrence Location"].unique()
data["Recurrence Location"].value_counts()
data.loc[:,"Recurrence Location"] = pd.Categorical(data["Recurrence Location"])

# Pie chart for Recurrence Location
values = data["Recurrence Location"].value_counts()
labels = data["Recurrence Location"].unique().tolist()
labels.remove(np.nan)
plt.pie(values,labels=labels, radius=1)
plt.show()
data.loc[:,"Recurrence Location"] = pd.Categorical(data["Recurrence Location"])

# percentage of each group
data["Recurrence Location"].value_counts()/data["Recurrence Location"].count()*100
data.dtypes

# Date of Recurrence
# convert dates to timestamp
data['Date of Recurrence'] = pd.to_datetime(data['Date of Recurrence'])
data['Date of Recurrence'] = data['Date of Recurrence'].astype('int64')
data['Date of Recurrence'] = data['Date of Recurrence'].replace(-9223372036854775808, np.nan)


data['Date of Last Known Alive'] = pd.to_datetime(data['Date of Last Known Alive'])
data['Date of Last Known Alive'] = data['Date of Last Known Alive'].astype('int64')
data['Date of Last Known Alive'] = data['Date of Last Known Alive'].replace(-9223372036854775808, np.nan)

# survival status
data.loc[:,"Survival Status"] = pd.Categorical(data["Survival Status"])

# date of death
data['Date of Death'] = pd.to_datetime(data['Date of Death'])
data['Date of Death'] = data['Date of Death'].astype('int64')
data['Date of Death'] = data['Date of Death'].replace(-9223372036854775808, np.nan)
