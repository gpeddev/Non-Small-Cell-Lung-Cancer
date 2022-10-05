
# ############################################################################################################## IMPORTS
from statistics import mean

from group_lasso import GroupLasso, LogisticGroupLasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, auc
from Pooling.Pooling import pooling
from SupportCode.Classification_Support import print_graph, model_evaluation, svm_model, feature_selection_lasso, \
    logistic_model
from SupportCode.Paths import CropTumor, CroppedWindow
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline

# ################################################################################## SELECT VARIABLES FOR REVIEWING CODE
latent_space = 256
baseline = True
model_path = "./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41"

# Use CropTumor for the first VAE(VAE-tumor), Use CroppedWindow for the second VAE(VAE-window)
#image_source = CropTumor
image_source = CroppedWindow


# ################################################################################################ LOAD AND PREPARE DATA
np.random.seed(0)
metric = "balanced_accuracy"

# Load clinical data
data = pd.read_csv("ClinicalDataAnalysis/data.csv")
data.reset_index()

# create new column names
column_names_vae1 = []
for i in range(latent_space * 2):
    column_names_vae1.append("VAE-"+str(i))

# load feature vectors from 5 kfold models
print("Pooling VAE")
pool_rslt = pooling(model_path + "/Models", latent_space, image_source)

# ############################################################################################ PERFORM FEATURE SELECTION

# START ############################ Create a list of dataframes with the case id the recurrence and the feature vectors
kfold_list_features = []
for i in range(5):
    temp1 = pd.DataFrame.from_dict(pool_rslt[i], orient='index', columns=column_names_vae1)
    selector = VarianceThreshold()
    temp1 = pd.DataFrame(selector.fit_transform(temp1), temp1.index, temp1.columns)
    temp1 = temp1.reset_index()
    temp1.rename(columns={"index": "Case ID"}, inplace=True)
# new_data => ready dataformat for use in machine learning algorithms
    new_data = data.loc[:, ["Case ID", "Recurrence"]]
    new_data = pd.merge(new_data, temp1)
    new_data["Recurrence"] = new_data["Recurrence"].replace("yes", 1)
    new_data["Recurrence"] = new_data["Recurrence"].replace("no", 0)
    kfold_list_features.append(new_data)
# kfold_list_features CASE ID, Recurrence, VAE feature vectors

# END ############################## Create a list of dataframes with the case id the recurrence and the feature vectors

# START ################################################################################################## Clinical data

temp = data
dataset_path = model_path + "/DatasetSplits"
patient_names_test_dataset = np.load(dataset_path + "/test_dataset_fold_" + str(i + 1) + ".npy")
# removes file extension
patient_names_test_dataset = [patient.rsplit(".")[0] for patient in patient_names_test_dataset]

temp["%GG"] = temp["%GG"].replace('0%', 0)
temp["%GG"] = temp["%GG"].replace('>0 - 25%', 1)
temp["%GG"] = temp["%GG"].replace('50 - 75%', 2)
temp["%GG"] = temp["%GG"].replace('25 - 50%', 3)
temp["%GG"] = temp["%GG"].replace('75 - < 100%', 4)
temp["%GG"] = temp["%GG"].replace('100%', 5)
temp.drop("Recurrence", inplace=True, axis=1)

dummies = ["Gender", "Ethnicity", "Smoking status", "Tumor Location (choice=RUL)",
           "Tumor Location (choice=RML)",
           "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)", "Tumor Location (choice=LLL)",
           "Tumor Location (choice=L Lingula)", "Histology", "Pathological T stage", "Pathological N stage",
           "Pathological M stage", "Histopathological Grade",
           "Pleural invasion (elastic, visceral, or parietal)",
           "Adjuvant Treatment", "Chemotherapy", "Radiation"]
clinical_data = pd.get_dummies(temp, columns=dummies)
# END #################################################################################################### Clinical data

# START ############################ BASELINE merge clinical_data AND kfold_list_features ##############################
final_data=[]

for i in range(5):
    if baseline is False:
        final_data.append(pd.merge(clinical_data, kfold_list_features[i]))
    else:
        final_data.append(kfold_list_features[i])

# final_data contains the result.
# final data when we have baseline contains
# Case ID, Recurrence, VAE-0 .. VAE-511
# final_data when not at baseline contains
# Case ID, clinical data.... , VAE-0 ... VAE-511

# END ############################## BASELINE merge clinical_data AND kfold_list_features ##############################

# START ############################# LASSO GROUP LASSO ################################################################
for i in range(5):
    if baseline is True:
        #final_data[i].drop("Recurrence", axis=1).drop()
        pipe = Pipeline(
            memory=None,
            steps=[
                (
                    "scaler", StandardScaler(),
                    "variable_selection",
                    LogisticGroupLasso(
                        groups=list(range(0, 512)),
                        group_reg=0.0005,
                        l1_reg=0,
                        scale_reg="inverse_group_size",
                        subsampling_scheme=1,
                        supress_warning=True,
                    ),
                ),
            ],
        )

        search = GridSearchCV(pipe,


        )

#        df.loc[:, 'C':'E']
        pipe.fit(final_data[i].loc[:, "VAE-0":"VAE-511"], final_data[i].loc[:, "Recurrence"])
        print(f"Number variables: {len(gl.sparsity_mask)}")
        print(f"Number of chosen variables: {gl.sparsity_mask.sum()}")
        break
    else:

        break
# END ############################### LASSO GROUP LASSO ################################################################