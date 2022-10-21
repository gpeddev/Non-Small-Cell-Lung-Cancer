from SupportCode.CancerRecurrence_Support import get_clinical_data, get_features_vectors, feature_selection_lasso, \
    merge_dataframes
from SupportCode.Paths import CropTumor, CroppedWindow
import pandas as pd
import numpy as np


latent_space = 256
baseline = False
model_path = "./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41"

# Use CropTumor for the first VAE(VAE-tumor), Use CroppedWindow for the second VAE(VAE-window)
#image_source = CropTumor
image_source = CroppedWindow


# ################################################################################################ LOAD AND PREPARE DATA
np.random.seed(0)

# load clinical data
clinical_data = get_clinical_data()
# clinical_data is a dataframe that contains the "Case ID" and "Recurrence"

# load feature vectors
vectorfeatures_list = get_features_vectors(latent_space, model_path, image_source)
# vectorfeatures_list is a list of feature vectors (dataframe) extracted from CT

final_dataframe = merge_dataframes(baseline, clinical_data, vectorfeatures_list)
# final_dataframe is a list of the final dataframes (either baseline or clinical data and baseline)

training_dataset, testing_dataset = feature_selection_lasso(clinical_data, vectorfeatures_list, model_path, baseline)