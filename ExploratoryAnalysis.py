import pandas as pd
import numpy as np
import os

# load dataframe
df = pd.read_csv("./ClinicalData/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")

# find patients in the study
patient_namelist = os.listdir("./Data/00_Images")

# removes .nii
patient_namelist = [patient.rsplit(".")[0] for patient in patient_namelist]

df = df[df['Case ID'].isin(patient_namelist)]

print(df.dtypes)

df.loc[:,"Patient affiliation"] = pd.Categorical(df["Patient affiliation"])


df.loc[:,'Weight (lbs)'] = pd.to_numeric(df['Weight (lbs)'], errors='coerce')


df.loc[:,"Gender"] = pd.Categorical(df["Gender"])


df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])


df.loc[:,"Smoking status"] = pd.Categorical(df["Smoking status"])


df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])



df.loc[:,"Pack Years"] = df["Pack Years"].replace("Not Collected",np.nan)
#df.loc[:,'Pack Years'] = pd.to_numeric(df['Pack Years'], downcast='integer', errors='coerce')
df.loc[:,'Pack Years'] = df['Pack Years'].astype('Int64')

print(df["Pack Years"])

df.loc[:,'Quit Smoking Year'] = df['Quit Smoking Year'].astype('Int64')

df.loc[:,"%GG"] = pd.Categorical(df["%GG"])

df.loc[:,"Tumor Location (choice=RUL)"] = pd.Categorical(df["Tumor Location (choice=RUL)"])
df.loc[:,"Tumor Location (choice=RML)"] = pd.Categorical(df["Tumor Location (choice=RML)"])
df.loc[:,"Tumor Location (choice=RLL)"] = pd.Categorical(df["Tumor Location (choice=RLL)"])
df.loc[:,"Tumor Location (choice=LUL)"] = pd.Categorical(df["Tumor Location (choice=LUL)"])
df.loc[:,"Tumor Location (choice=LLL)"] = pd.Categorical(df["Tumor Location (choice=LLL)"])
df.loc[:,"Tumor Location (choice=L Lingula)"] = pd.Categorical(df["Tumor Location (choice=L Lingula)"])
df.loc[:,"Tumor Location (choice=Unknown)"] = pd.Categorical(df["Tumor Location (choice=Unknown)"])
print(df.dtypes)

df.to_csv("data.csv",index=False)