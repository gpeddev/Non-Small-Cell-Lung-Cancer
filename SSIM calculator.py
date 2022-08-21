import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# load models
from SupportCode.datasets_support import preprocess_data
from SupportCode.Paths import CropTumor,CroppedWindow
from skimage.metrics import structural_similarity as ssim

# point to the model you want to calculate SSIM score
# if you want to test VAE 2 category use CroppedWindow else use CropTumor for VAE 1
# for testing VAE on validation dataset set val=True or for testing dataset set val = False
model_path="./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44"

def evaluate_vae_ssim(vae_path,image_path,val):
    """evaluate vaes ssim."""
    if val is True:
        # load names of test datasets for each model
        val_dataset_1 = np.load(vae_path + "/DatasetSplits/val_dataset_fold_1.npy")
        val_dataset_2 = np.load(vae_path + "/DatasetSplits/val_dataset_fold_2.npy")
        val_dataset_3 = np.load(vae_path + "/DatasetSplits/val_dataset_fold_3.npy")
        val_dataset_4 = np.load(vae_path + "/DatasetSplits/val_dataset_fold_4.npy")
        val_dataset_5 = np.load(vae_path + "/DatasetSplits/val_dataset_fold_5.npy")
        datasets = [val_dataset_1, val_dataset_2, val_dataset_3, val_dataset_4, val_dataset_5]
    else:
        test_dataset_1 = np.load(vae_path + "/DatasetSplits/test_dataset_fold_1.npy")
        test_dataset_2 = np.load(vae_path + "/DatasetSplits/test_dataset_fold_2.npy")
        test_dataset_3 = np.load(vae_path + "/DatasetSplits/test_dataset_fold_3.npy")
        test_dataset_4 = np.load(vae_path + "/DatasetSplits/test_dataset_fold_4.npy")
        test_dataset_5 = np.load(vae_path + "/DatasetSplits/test_dataset_fold_5.npy")
        datasets = [test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5]

    vae_1 = tf.keras.models.load_model(vae_path + '/Models/VAE_1.h5', compile=False)
    vae_2 = tf.keras.models.load_model(vae_path + '/Models/VAE_2.h5', compile=False)
    vae_3 = tf.keras.models.load_model(vae_path + '/Models/VAE_3.h5', compile=False)
    vae_4 = tf.keras.models.load_model(vae_path + '/Models/VAE_4.h5', compile=False)
    vae_5 = tf.keras.models.load_model(vae_path + '/Models/VAE_5.h5', compile=False)
    VAEs = [vae_1, vae_2, vae_3, vae_4, vae_5]


    results = []
    for i in range(5):
        s_s_i_m = 0
        data = preprocess_data(image_path, datasets[i])
        for item in data:
            temp = np.reshape(item, newshape=(1, item.shape[0], item.shape[1], 1))
            result = VAEs[i](temp)
            input_image = np.reshape(item, newshape=(item.shape[0], item.shape[1]))
            output_image = np.reshape(result[0], newshape=(result[0].shape[0], result[0].shape[1]))
            s_s_i_m = s_s_i_m + ssim(input_image, output_image)
        results.append(s_s_i_m / data.shape[0])
    return sum(results)/5


# my computer couldnt more than one since each evaluate_vae_ssim loads 5 models so I had to run it one by one
x1=evaluate_vae_ssim(model_path,
                     image_path=CropTumor,
                     val=False)
print(x1)