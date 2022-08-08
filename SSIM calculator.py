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
    vae_1 = tf.keras.models.load_model(vae_path + '/VAE_1.h5', compile=False)
    vae_2 = tf.keras.models.load_model(vae_path + '/VAE_2.h5', compile=False)
    vae_3 = tf.keras.models.load_model(vae_path + '/VAE_3.h5', compile=False)
    vae_4 = tf.keras.models.load_model(vae_path + '/VAE_4.h5', compile=False)
    vae_5 = tf.keras.models.load_model(vae_path + '/VAE_5.h5', compile=False)
    VAEs = [vae_1, vae_2, vae_3, vae_4, vae_5]

    # load names of test datasets for each model
    test_dataset_1 = np.load(dataset_path+"/test_dataset_fold_1.npy")
    test_dataset_2 = np.load(dataset_path+"/test_dataset_fold_2.npy")
    test_dataset_3 = np.load(dataset_path+"/test_dataset_fold_3.npy")
    test_dataset_4 = np.load(dataset_path+"/test_dataset_fold_4.npy")
    test_dataset_5 = np.load(dataset_path+"/test_dataset_fold_5.npy")
    test_datasets = [test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5]

    results = []
    for i in range(5):
        s_s_i_m = 0
        data = preprocess_data(CropTumor, test_datasets[i])
        for item in data:
            temp = np.reshape(item, newshape=(1, item.shape[0], item.shape[1], 1))
            result = VAEs[i](temp)
            input_image = np.reshape(item, newshape=(item.shape[0], item.shape[1]))
            input_image = input_image
            output_image = np.reshape(result[0], newshape=(result[0].shape[0], result[0].shape[1]))
            output_image = output_image
            s_s_i_m = s_s_i_m + ssim(input_image, output_image)
        results.append(s_s_i_m / data.shape[0])
    return sum(results)/5


#x=evaluate_vae_ssim("./test_directory/instance5/Models","./test_directory/instance5/DatasetSplits")



x=evaluate_vae_ssim("./test_directory/output_desktop_1/Models","./test_directory/output_desktop_1/DatasetSplits")
print("output_desktop_1")
print(x)

x=evaluate_vae_ssim("./test_directory/output_desktop_2/Models","./test_directory/output_desktop_2/DatasetSplits")
print("output_desktop_2")
print(x)

x=evaluate_vae_ssim("./test_directory/instance1/Models","./test_directory/instance1/DatasetSplits")
print("instance1")
print(x)

x=evaluate_vae_ssim("./test_directory/instance2/Models","./test_directory/instance2/DatasetSplits")
print("instance2")
print(x)

x=evaluate_vae_ssim("./test_directory/instance3/Models","./test_directory/instance3/DatasetSplits")
print("instance3")
print(x)

x=evaluate_vae_ssim("./test_directory/instance4/Models","./test_directory/instance4/DatasetSplits")
print("instance4")
print(x)

x=evaluate_vae_ssim("./test_directory/instance5/Models","./test_directory/instance5/DatasetSplits")
print("instance5")
print(x)

x=evaluate_vae_ssim("./test_directory/instance6/Models","./test_directory/instance6/DatasetSplits")
print("instance6")
print(x)

x=evaluate_vae_ssim("./test_directory/instance7/Models","./test_directory/instance7/DatasetSplits")
print("instance7")
print(x)

x=evaluate_vae_ssim("./test_directory/instance8/Models","./test_directory/instance8/DatasetSplits")
print("instance8")
print(x)


x=evaluate_vae_ssim("./test_directory/instance9/Models","./test_directory/instance9/DatasetSplits")
print("instance9")
print(x)


x=evaluate_vae_ssim("./test_directory/instance10/Models","./test_directory/instance10/DatasetSplits")
print("instance10")
print(x)


