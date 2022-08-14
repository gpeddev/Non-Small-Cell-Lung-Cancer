import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# load models
from SupportCode.datasets_support import preprocess_data
from SupportCode.Paths import CropTumor
from skimage.metrics import structural_similarity as ssim


def evaluate_vae_ssim(vae_path, dataset_path):
    """evaluate vaes ssim."""

        # load names of test datasets for each model
    val_dataset_1 = np.load(dataset_path + "/val_dataset_fold_1.npy")
    val_dataset_2 = np.load(dataset_path + "/val_dataset_fold_2.npy")
    val_dataset_3 = np.load(dataset_path + "/val_dataset_fold_3.npy")
    val_dataset_4 = np.load(dataset_path + "/val_dataset_fold_4.npy")
    val_dataset_5 = np.load(dataset_path + "/val_dataset_fold_5.npy")
    datasets = [val_dataset_1, val_dataset_2, val_dataset_3, val_dataset_4, val_dataset_5]



    vae_1 = tf.keras.models.load_model(vae_path + '/VAE_1.h5', compile=False)
    vae_2 = tf.keras.models.load_model(vae_path + '/VAE_2.h5', compile=False)
    vae_3 = tf.keras.models.load_model(vae_path + '/VAE_3.h5', compile=False)
    vae_4 = tf.keras.models.load_model(vae_path + '/VAE_4.h5', compile=False)
    vae_5 = tf.keras.models.load_model(vae_path + '/VAE_5.h5', compile=False)
    VAEs = [vae_1, vae_2, vae_3, vae_4, vae_5]



    results = []
    for i in range(5):
        s_s_i_m = 0
        data = preprocess_data(CropTumor, datasets[i])
        for item in data:
            temp = np.reshape(item, newshape=(1, item.shape[0], item.shape[1], 1))
            result = VAEs[i](temp)
            input_image = np.reshape(item, newshape=(item.shape[0], item.shape[1]))
            output_image = np.reshape(result[0], newshape=(result[0].shape[0], result[0].shape[1]))
            s_s_i_m = s_s_i_m + ssim(input_image, output_image)
        results.append(s_s_i_m / data.shape[0])
    return sum(results)/5


# Model 1



x=evaluate_vae_ssim("./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44/Models",
                    "./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44/DatasetSplits",
                    dataset_to_test="val")
print("model_1 kl=1")
print(x)

x=evaluate_vae_ssim("./BestResults/VAE_1/Model_1/18_2022-08-10_01_50_29/Models",
                    "./BestResults/VAE_1/Model_1/18_2022-08-10_01_50_29/DatasetSplits",
                    dataset_to_test="val")
print("model_1 kl=0.5")
print(x)

x=evaluate_vae_ssim("./BestResults/VAE_1/Model_1/19_2022-08-10_01_53_19/Models",
                    "./BestResults/VAE_1/Model_1/19_2022-08-10_01_53_19/DatasetSplits",
                    dataset_to_test="val")
print("model_1 kl=0.25")
print(x)

x=evaluate_vae_ssim("./BestResults/VAE_1/Model_1/20_2022-08-10_01_55_25/Models",
                    "./BestResults/VAE_1/Model_1/20_2022-08-10_01_55_25/DatasetSplits",
                    dataset_to_test="val")
print("model_1 kl=0.125")
print(x)

