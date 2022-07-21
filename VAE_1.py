########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import SimpleITK as Sitk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import tensorboard as tb
from Models.VAE_1.VAE_1_mse import VAE, preprocess, early_stopping, tensorboard_callback
from Paths import CropTumor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Helpers functions
# Returns an array with all the slices from the file_list CTs
def get_all_slices(filepath, file_list):
    results = []
    for item in file_list:
        image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath+item, Sitk.sitkUInt8))
        for i in range(image_data.shape[0]):
            results.append(image_data[i, :, :])
    return results


# Make initial split
file_array = np.array(os.listdir(CropTumor))
# init_train is the part used in kfold cross validation
# init_test is the part used for testing
train_filename_list, holdout_filename_list = train_test_split(file_array,
                                                              test_size=0.10,
                                                              shuffle=True)

# Store initial model weights
initial_weights = VAE.get_weights()

kFold = KFold(n_splits=5, shuffle=True, random_state=1)
kFold_results = []

for train, val in kFold.split(train_filename_list):

    # build train and valuation datasets
    train_slices = get_all_slices(CropTumor, train_filename_list[train])
    train_db = tf.data.Dataset.from_tensor_slices(train_slices)\
        .map(preprocess)\
        .batch(16)\
        .prefetch(tf.data.AUTOTUNE)\
        .shuffle(int(10e3))

    val_slices = get_all_slices(CropTumor, train_filename_list[val])
    val_db = tf.data.Dataset.from_tensor_slices(val_slices)\
        .map(preprocess)\
        .batch(16)\
        .prefetch(tf.data.AUTOTUNE)

    # reset model weights
    VAE.set_weights(initial_weights)
    # fit model
    fit_results = VAE.fit(train_db,
                          epochs=500,
                          validation_data=val_db,
                          callbacks=[early_stopping, tensorboard_callback],
                          verbose=1)
    kFold_results.append(fit_results)

plt.plot(kFold_results.history['loss'], label='MAE (training data)')
plt.plot(kFold_results.history['val_loss'], label='MAE (validation data)')
plt.title('VAE for mse')
plt.ylabel('Error')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
