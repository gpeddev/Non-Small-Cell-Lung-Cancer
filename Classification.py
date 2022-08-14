#       Load pooling
from Pooling.Pooling import pooling
from SupportCode.Paths import CropTumor
import pandas as pd

vae_1_latent_space=256
# Load clinical data
data = pd.read_csv("ClinicalDataAnalysis/data.csv")
data.reset_index()
# create new column names
column_names_vae1=[]
for i in range(vae_1_latent_space*2):
    column_names_vae1.append("VAE1_"+str(i))
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


