import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None

# load dataframe
messy_data = pd.read_csv("./ClinicalData/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")

# find patients in the study
patient_namelist = os.listdir("./Data/00_Images")

# removes .nii
patient_namelist = [patient.rsplit(".")[0] for patient in patient_namelist]

# keeps patients of interest
df = messy_data[messy_data['Case ID'].isin(patient_namelist)]
df.dtypes
df_description=df.describe(include='all')
df.corr()[['Age at Histological Diagnosis']].sort_values(by='Age at Histological Diagnosis', ascending=False)
df.isnull().sum()

df_correlation=df.corr()

sns.heatmap(df.corr())
sns.heatmap(df.corr());


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()["Recurrence"].sort_values(by="Recurrence", ascending=False),vmin=-1, vmax=1, annot=True, cmap='BrBG')

plt.show()



























# ############################################################################### Convert variables to appropriate type

print(df.dtypes)
# Convert categorical variables to categorical variables
df.loc[:,"Patient affiliation"] = pd.Categorical(df["Patient affiliation"])
df.loc[:,"Gender"] = pd.Categorical(df["Gender"])
df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])
df.loc[:,"Smoking status"] = pd.Categorical(df["Smoking status"])
df.loc[:,"Ethnicity"] = pd.Categorical(df["Ethnicity"])
df.loc[:,"%GG"] = pd.Categorical(df["%GG"])
df.loc[:,"Tumor Location (choice=RUL)"] = pd.Categorical(df["Tumor Location (choice=RUL)"])
df.loc[:,"Tumor Location (choice=RML)"] = pd.Categorical(df["Tumor Location (choice=RML)"])
df.loc[:,"Tumor Location (choice=RLL)"] = pd.Categorical(df["Tumor Location (choice=RLL)"])
df.loc[:,"Tumor Location (choice=LUL)"] = pd.Categorical(df["Tumor Location (choice=LUL)"])
df.loc[:,"Tumor Location (choice=LLL)"] = pd.Categorical(df["Tumor Location (choice=LLL)"])
df.loc[:,"Tumor Location (choice=L Lingula)"] = pd.Categorical(df["Tumor Location (choice=L Lingula)"])
df.loc[:,"Tumor Location (choice=Unknown)"] = pd.Categorical(df["Tumor Location (choice=Unknown)"])
df.loc[:,"Histology"]=pd.Categorical(df["Histology"])
df.loc[:,"Pathological T stage"]=pd.Categorical(df["Pathological T stage"])
df.loc[:,"Pathological N stage"]=pd.Categorical(df["Pathological N stage"])
df.loc[:,"Pathological M stage"]=pd.Categorical(df["Pathological M stage"])
df.loc[:,"Histopathological Grade"]=pd.Categorical(df["Histopathological Grade"])
df.loc[:,"Lymphovascular invasion"]=pd.Categorical(df["Lymphovascular invasion"])
df.loc[:,"Pleural invasion (elastic, visceral, or parietal)"]=pd.Categorical(df["Pleural invasion (elastic, visceral, or parietal)"])
df.loc[:,"EGFR mutation status"]=pd.Categorical(df["EGFR mutation status"])
df.loc[:,"KRAS mutation status"]=pd.Categorical(df["KRAS mutation status"])
df.loc[:,"ALK translocation status"]=pd.Categorical(df["ALK translocation status"])
df.loc[:,"Adjuvant Treatment"]=pd.Categorical(df["Adjuvant Treatment"])
df.loc[:,"Chemotherapy"]=pd.Categorical(df["Chemotherapy"])
df.loc[:,"Radiation"]=pd.Categorical(df["Radiation"])
df.loc[:,"Recurrence"]=pd.Categorical(df["Recurrence"])
df.loc[:,"Recurrence Location"]=pd.Categorical(df["Recurrence Location"])
df.loc[:,"Survival Status"]=pd.Categorical(df["Survival Status"])
# replace not collected with nan
df.loc[:,"Lymphovascular invasion"] = pd.Categorical(df["Lymphovascular invasion"])
df.loc[:,"Lymphovascular invasion"] = df["Lymphovascular invasion"].replace("Not Collected",np.nan)


# convert dates to timestamp
df['Date of Recurrence'] = pd.to_datetime(df['Date of Recurrence'])
df['Date of Recurrence'] = df['Date of Recurrence'].astype('int64')
df['Date of Recurrence'] = df['Date of Recurrence'].replace(-9223372036854775808, np.nan)

df['Date of Last Known Alive'] = pd.to_datetime(df['Date of Last Known Alive'])
df['Date of Last Known Alive'] = df['Date of Last Known Alive'].astype('int64')
df['Date of Last Known Alive'] = df['Date of Last Known Alive'].replace(-9223372036854775808, np.nan)

df['Date of Death'] = pd.to_datetime(df['Date of Death'])
df["Date of Death"] = df['Date of Death'].astype('int64')
df['Date of Death'] = df['Date of Death'].replace(-9223372036854775808, np.nan)

df['CT Date'] = pd.to_datetime(df['CT Date'])
df["CT Date"] = df['CT Date'].astype('int64')
df['CT Date'] = df['CT Date'].replace(-9223372036854775808, np.nan)

df['PET Date'] = pd.to_datetime(df['PET Date'])
df["PET Date"] = df['PET Date'].astype('int64')
df['PET Date'] = df['PET Date'].replace(-9223372036854775808, np.nan)


# Convert ordinal categorical variables to numbers 1,2,3 etc
df.loc[:,"%GG"] = df["%GG"].replace("0%",0)
df.loc[:,"%GG"] = df["%GG"].replace(">0 - 25%",1)
df.loc[:,"%GG"] = df["%GG"].replace("25 - 50%",2)
df.loc[:,"%GG"] = df["%GG"].replace("50 - 75%",3)
df.loc[:,"%GG"] = df["%GG"].replace("75 - < 100%",4)

# Convert numerical numbers to int/float
df.loc[:,'Weight (lbs)'] = pd.to_numeric(df['Weight (lbs)'], errors='coerce')
df.loc[:,'Quit Smoking Year'] = df['Quit Smoking Year'].astype('Int64')
df.loc[:,'Time to Death (days)'] = df['Time to Death (days)'].astype('Int64')
df.loc[:,'Days between CT and surgery'] = df['Days between CT and surgery'].astype('Int64')

df.loc[:,"Pack Years"] = df["Pack Years"].replace("Not Collected",np.nan)
df.loc[:,'Pack Years'] = df['Pack Years'].astype('Int64')


# convert dummy variables
df = pd.get_dummies(df, prefix='Patient affiliation_', columns=['Patient affiliation'])
df = pd.get_dummies(df, prefix='Gender_', columns=['Gender'])
df = pd.get_dummies(df, prefix='Ethnicity_', columns=['Ethnicity'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=RUL)_', columns=['Tumor Location (choice=RUL)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=RML)_', columns=['Tumor Location (choice=RML)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=RLL)_', columns=['Tumor Location (choice=RLL)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=LUL)_', columns=['Tumor Location (choice=LUL)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=LLL)_', columns=['Tumor Location (choice=LLL)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=L Lingula)_', columns=['Tumor Location (choice=L Lingula)'])
df = pd.get_dummies(df, prefix='Tumor Location (choice=Unknown)_', columns=['Tumor Location (choice=Unknown)'])
df = pd.get_dummies(df, prefix='Histology_', columns=['Histology'])
df = pd.get_dummies(df, prefix='Pathological T stage_', columns=['Pathological T stage'])
df = pd.get_dummies(df, prefix='Pathological N stage', columns=['Pathological N stage'])
df = pd.get_dummies(df, prefix='Pathological M stage_', columns=['Pathological M stage'])
df = pd.get_dummies(df, prefix='Histopathological Grade_', columns=['Histopathological Grade'])
df = pd.get_dummies(df, prefix='Lymphovascular invasion_', columns=['Lymphovascular invasion'])
df = pd.get_dummies(df, prefix='Pleural invasion (elastic, visceral, or parietal)_', columns=['Pleural invasion (elastic, visceral, or parietal)'])
df = pd.get_dummies(df, prefix='EGFR mutation status_', columns=['EGFR mutation status'])
df = pd.get_dummies(df, prefix='KRAS mutation status_', columns=['KRAS mutation status'])
df = pd.get_dummies(df, prefix='ALK translocation status_', columns=['ALK translocation status'])
df = pd.get_dummies(df, prefix='Adjuvant Treatment_', columns=['Adjuvant Treatment'])
df = pd.get_dummies(df, prefix='Chemotherapy_', columns=['Chemotherapy'])
df = pd.get_dummies(df, prefix='Radiation_', columns=['Radiation'])
df = pd.get_dummies(df, prefix='Recurrence_', columns=['Recurrence'])
df = pd.get_dummies(df, prefix='Recurrence Location_', columns=['Recurrence Location'])
df = pd.get_dummies(df, prefix='Survival Status_', columns=['Survival Status'])


#df.to_csv("data.csv",index=False)