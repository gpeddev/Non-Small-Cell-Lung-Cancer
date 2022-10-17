# ############################################################################################################## IMPORTS
from SupportCode.Classification_Support import print_graph, model_evaluation
from SupportCode.FeatureSelection_Classificaiton_Support import apply_SVM, apply_group_lasso, get_data, \
    feature_selection_lasso
from SupportCode.Paths import CropTumor, CroppedWindow
import numpy as np

# ################################################################################## SELECT VARIABLES FOR REVIEWING CODE
latent_space = 256
baseline = True
model_path = "./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44"

# Use CropTumor for the first VAE(VAE-tumor), Use CroppedWindow for the second VAE(VAE-window)
image_source = CropTumor
#image_source = CroppedWindow


# ################################################################################################ LOAD AND PREPARE DATA
np.random.seed(0)
metric = "balanced_accuracy"


final_data=get_data(baseline,latent_space,model_path,image_source)

reduced_columns=apply_group_lasso(model_path, baseline, final_data)
#reduced_columns=feature_selection_lasso(final_data)

grid_models, testing_dataset=apply_SVM(model_path,final_data,reduced_columns)

model_evaluation(grid_models, testing_dataset)

print_graph(grid_models, testing_dataset)
