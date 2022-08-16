#       Load pooling
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Pooling.Pooling import pooling
from SupportCode.Paths import CropTumor
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from group_lasso.utils import extract_ohe_groups
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from group_lasso import LogisticGroupLasso
np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True


vae_1_latent_space=256
# Load clinical data
data = pd.read_csv("ClinicalDataAnalysis/data.csv")
data.reset_index()
# create new column names
column_names_vae1=[]
for i in range(vae_1_latent_space*2):
    column_names_vae1.append("VAE1-"+str(i))
# load vectors from 5 kfold models
rslt=pooling("./BestResults/VAE_1/Model_1/19_2022-08-10_01_53_19/Models", vae_1_latent_space,CropTumor)

# create pandas from dictionary
kfold_list_features=[]
for i in range(5):
    temp=pd.DataFrame.from_dict(rslt[i], orient='index', columns=column_names_vae1)
    temp=temp.reset_index()
    temp.rename(columns={"index": "Case ID"}, inplace=True)
    kfold_list_features.append(temp)



################################ work with one to begin with
new_data=pd.merge(data,kfold_list_features[0])

##################################################################### LASSO

new_data["%GG"].unique()
new_data["%GG"]=new_data["%GG"].replace('0%',0)
new_data["%GG"]=new_data["%GG"].replace('>0 - 25%',1)
new_data["%GG"]=new_data["%GG"].replace('50 - 75%',2)
new_data["%GG"]=new_data["%GG"].replace('25 - 50%',3)
new_data["%GG"]=new_data["%GG"].replace('75 - < 100%',4)
new_data["%GG"]=new_data["%GG"].replace('100%',5)
new_data["%GG"].unique()

new_data["Recurrence"]=new_data["Recurrence"].replace("yes",1)
new_data["Recurrence"]=new_data["Recurrence"].replace("no",0)


new_data.drop("Case ID",inplace=True,axis=1)
dummies=["Gender","Ethnicity", "Smoking status", "Tumor Location (choice=RUL)", "Tumor Location (choice=RML)",
         "Tumor Location (choice=RLL)","Tumor Location (choice=LUL)","Tumor Location (choice=LLL)",
         "Tumor Location (choice=L Lingula)","Histology","Pathological T stage","Pathological N stage",
         "Pathological M stage","Histopathological Grade","Pleural invasion (elastic, visceral, or parietal)",
         "Adjuvant Treatment","Chemotherapy","Radiation"]
new_data["Pathological T stage"].unique()
new_data["Pathological N stage"].unique()

y=new_data["Recurrence"]
x=new_data.drop("Recurrence",axis=1)


ohe = OneHotEncoder()
onehot_data = ohe.fit_transform(data[dummies])
groups = extract_ohe_groups(ohe)

set(list(x.columns))-set(dummies)

X = scipy.sparse.hstack([onehot_data,scipy.sparse.csr_matrix(new_data[num_columns])])

