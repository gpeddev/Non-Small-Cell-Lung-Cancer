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
print(df.dtypes)

df.loc[:,'Weight (lbs)'] = pd.to_numeric(df['Weight (lbs)'], errors='coerce')
print(df.dtypes)

df.loc[:,"Gender"] = pd.Categorical(df["Gender"])
print(df.dtypes)

df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])
print(df.dtypes)

df.loc[:,"Smoking status"] = pd.Categorical(df["Smoking status"])
print(df.dtypes)

df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])
print(df.dtypes)

df.loc[:,'Pack Years'] = pd.to_numeric(df['Pack Years'], downcast='integer', errors='coerce')
print(df.dtypes)
