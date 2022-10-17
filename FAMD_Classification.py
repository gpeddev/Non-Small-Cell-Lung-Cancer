# ############################################################################################################## IMPORTS
from SupportCode.FAMD_Classification_Support import get_data
from SupportCode.Paths import CropTumor, CroppedWindow
import numpy as np
import prince

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

data=get_data(baseline,latent_space,model_path,image_source)
# baseline=True, data => Case ID, Recurrence, VAE-0... VAE-511
# baseline=False, data => Case ID, Clinical Data... Clinical Data ,Recurrence, VAE-0... VAE-511
print("Test")

for i in range(5):
    temp=data[i].drop("Case ID",axis=1)
    pca = prince.PCA(
        n_components=5,
        n_iter=3,
        rescale_with_mean=True,
        rescale_with_std=True,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42)
    pca = pca.fit(temp.drop("Recurrence",axis=1))
