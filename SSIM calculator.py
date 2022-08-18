import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# load models
from SupportCode.datasets_support import preprocess_data
from SupportCode.Paths import CropTumor,CroppedWindow
from skimage.metrics import structural_similarity as ssim


def evaluate_vae_ssim(vae_path, dataset_path,image_path,val):
    """evaluate vaes ssim."""
    if val==True:
        # load names of test datasets for each model
        val_dataset_1 = np.load(dataset_path + "/val_dataset_fold_1.npy")
        val_dataset_2 = np.load(dataset_path + "/val_dataset_fold_2.npy")
        val_dataset_3 = np.load(dataset_path + "/val_dataset_fold_3.npy")
        val_dataset_4 = np.load(dataset_path + "/val_dataset_fold_4.npy")
        val_dataset_5 = np.load(dataset_path + "/val_dataset_fold_5.npy")
        datasets = [val_dataset_1, val_dataset_2, val_dataset_3, val_dataset_4, val_dataset_5]
    else:
        test_dataset_1 = np.load(dataset_path + "/test_dataset_fold_1.npy")
        test_dataset_2 = np.load(dataset_path + "/test_dataset_fold_2.npy")
        test_dataset_3 = np.load(dataset_path + "/test_dataset_fold_3.npy")
        test_dataset_4 = np.load(dataset_path + "/test_dataset_fold_4.npy")
        test_dataset_5 = np.load(dataset_path + "/test_dataset_fold_5.npy")
        datasets = [test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5]



    vae_1 = tf.keras.models.load_model(vae_path + '/VAE_1.h5', compile=False)
    vae_2 = tf.keras.models.load_model(vae_path + '/VAE_2.h5', compile=False)
    vae_3 = tf.keras.models.load_model(vae_path + '/VAE_3.h5', compile=False)
    vae_4 = tf.keras.models.load_model(vae_path + '/VAE_4.h5', compile=False)
    vae_5 = tf.keras.models.load_model(vae_path + '/VAE_5.h5', compile=False)
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


# my computer couldnt handle all of them. Run out of memory. Uncomment if you want to run all together.


x1=evaluate_vae_ssim("./BestResults/VAE_2/Model_1/17_2022-08-15_19_47_42/Models",
                     "./BestResults/VAE_2/Model_1/17_2022-08-15_19_47_42/DatasetSplits",
                     image_path=CroppedWindow,
                     val=True)
print("model_1 kl=1")
print(x1)

x1=evaluate_vae_ssim("./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41/Models",
                     "./BestResults/VAE_2/Model_1/20_2022-08-15_19_51_41/DatasetSplits",
                     image_path=CroppedWindow,
                     val=False)
print("model_1 kl=0.125")
print(x1)

x1=evaluate_vae_ssim("./BestResults/VAE_2/Model_1/22_2022-08-16_04_02_50/Models",
                     "./BestResults/VAE_2/Model_1/22_2022-08-16_04_02_50/DatasetSplits",
                     image_path=CroppedWindow,
                     val=True)
print("model_1 kl=0.5")
print(x1)

x1=evaluate_vae_ssim("./BestResults/VAE_2/Model_1/23_2022-08-15_19_35_09/Models",
                     "./BestResults/VAE_2/Model_1/23_2022-08-15_19_35_09/DatasetSplits",
                     image_path=CroppedWindow,
                     val=True)
print("model_1 kl=0.25")
print(x1)