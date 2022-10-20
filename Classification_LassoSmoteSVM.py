
# ############################################################################################################## IMPORTS
from statistics import mean
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
baseline = False
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
    column_names_vae1.append("VAE1-"+str(i))

# load feature vectors from 5 kfold models
print("Pooling VAE")
pool_rslt = pooling(model_path + "/Models", latent_space, image_source)

# ############################################################################################ PERFORM FEATURE SELECTION
f1_score_result = []
number_of_variables_result = []

# Create a list of dataframes with the case id and the feature vectors
kfold_list_features = []
for i in range(5):
    temp1 = pd.DataFrame.from_dict(pool_rslt[i], orient='index', columns=column_names_vae1)
    selector = VarianceThreshold()
    temp1 = pd.DataFrame(selector.fit_transform(temp1), temp1.index, temp1.columns)
    temp1 = temp1.reset_index()
    temp1.rename(columns={"index": "Case ID"}, inplace=True)
    kfold_list_features.append(temp1)

# ############################################################################################## feature selection lasso
training_dataset, testing_dataset = feature_selection_lasso(data, kfold_list_features, model_path)

# ################################################################################################################## SVM
grid_models = svm_model(baseline, training_dataset, data, model_path)

# ###################################################################### MODEL EVALUATION AND PRINT STATISTICS AND PLOTS
model_evaluation(grid_models, testing_dataset, baseline)

# ################################################################################################################ GRAPH
print_graph(grid_models, testing_dataset)
