import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# load models
from SupportCode.datasets_support import preprocess_data
from SupportCode.Paths import CropTumor
from skimage.metrics import structural_similarity as ssim


def evaluate_vae_ssim(vae_path, dataset_path,val=True):
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
        data = preprocess_data(CropTumor, datasets[i])
        for item in data:
            temp = np.reshape(item, newshape=(1, item.shape[0], item.shape[1], 1))
            result = VAEs[i](temp)
            input_image = np.reshape(item, newshape=(item.shape[0], item.shape[1]))
            output_image = np.reshape(result[0], newshape=(result[0].shape[0], result[0].shape[1]))
            s_s_i_m = s_s_i_m + ssim(input_image, output_image)
        results.append(s_s_i_m / data.shape[0])
    return sum(results)/5


# my computer couldnt handle all of them. Run out of memory. Uncomment if you want to run all together.

# x1=evaluate_vae_ssim("./BestResults/VAE_1/Model_1a/17_2022-08-13_10_26_21/Models",
#                     "./BestResults/VAE_1/Model_1a/17_2022-08-13_10_26_21/DatasetSplits", val=True)
# print("model_1 kl=1")
# print(x1)
#
# x2=evaluate_vae_ssim("./BestResults/VAE_1/Model_1a/18_2022-08-13_10_27_30/Models",
#                     "./BestResults/VAE_1/Model_1a/18_2022-08-13_10_27_30/DatasetSplits", val=True)
# print("model_1 kl=0.5")
# print(x2)

# x3=evaluate_vae_ssim("./BestResults/VAE_1/Model_1a/19_2022-08-13_10_28_14/Models",
#                     "./BestResults/VAE_1/Model_1a/19_2022-08-13_10_28_14/DatasetSplits", val=True)
# print("model_1 kl=0.25")
# print(x3)
#
# x4=evaluate_vae_ssim("./BestResults/VAE_1/Model_1a/44_2022-08-14_11_53_46/Models",
#                     "./BestResults/VAE_1/Model_1a/44_2022-08-14_11_53_46/DatasetSplits", val=True)
# print("model_1 kl=0.125")
# print(x4)


x1=evaluate_vae_ssim("./BestResults/VAE_1/Model_1a/17_2022-08-13_10_26_21/Models",
                     "./BestResults/VAE_1/Model_1a/17_2022-08-13_10_26_21/DatasetSplits", val=False)
print("model_1 kl=1")
print(x1)

x1=evaluate_vae_ssim("./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44/Models",
                     "./BestResults/VAE_1/Model_1/17_2022-08-10_01_47_44/DatasetSplits", val=False)
print("model_1 kl=1")
print(x1)