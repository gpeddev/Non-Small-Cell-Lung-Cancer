########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import SimpleITK as Sitk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

from Models.VAE_1.VAE_1_parameters import batch_size
from Models.VAE_1.VAE_model_1 import VAE, early_stopping_kfold, tensorboard_callback, early_stopping_training_db
from Paths import CropTumor
import numpy as np


# Helpers functions
# Returns an array with all the slices from the file_list CTs
def get_all_slices(filepath, file_list):
    results = []
    for filename in file_list:
        image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath+filename, Sitk.sitkUInt8))
        for i in range(image_data.shape[0]):
            results.append(image_data[i, :, :])
    return np.array(results)


# Make initial split
file_array = np.array(os.listdir(CropTumor))
# init_train is the part used in kfold cross validation
# init_test is the part used for testing
train_filename_list, holdout_filename_list = train_test_split(file_array,
                                                              test_size=0.10,
                                                              shuffle=True)

# Store initial model weights
initial_weights = VAE.get_weights()

# ############################################################################################### KFOLD CROSS VALIDATION

kFold = KFold(n_splits=5, shuffle=True, random_state=1)
kFold_results = []

for train, val in kFold.split(train_filename_list):

    # build train and valuation datasets
    train_slices = get_all_slices(CropTumor, train_filename_list[train])
    val_slices = get_all_slices(CropTumor, train_filename_list[val])

    # adjust to 0 - 1 range
    train_slices = train_slices.astype("float32") / 255.0
    val_slices = val_slices.astype("float32") / 255.0

    # shuffle slices so each batch doesn't contain alot of slices from the same patient
    np.random.shuffle(train_slices)
    np.random.shuffle(val_slices)

    # reshape to match expected input
    train_slices = np.reshape(train_slices,
                              newshape=(train_slices.shape[0], train_slices.shape[1], train_slices.shape[2], 1))
    val_slices = np.reshape(val_slices,
                            newshape=(val_slices.shape[0], val_slices.shape[1], val_slices.shape[2], 1))

    # reset model weights before training
    VAE.set_weights(initial_weights)

    # fit model
    fit_results = VAE.fit(train_slices, train_slices,
                          epochs=1000,
                          validation_data=(val_slices, val_slices),
                          batch_size=batch_size,
                          callbacks=[early_stopping_kfold, tensorboard_callback],
                          verbose=1)
    kFold_results.append(fit_results)


# ######################################################################################## AVERAGE VALIDATION FROM KFOLD
add_val_losses = 0
for item in kFold_results:
    add_val_losses = add_val_losses + item.history["val_loss"][-1]

print("Average Loss at kfold at valuation data is: ")
print(add_val_losses / len(kFold_results))


# ###################################################### TRAIN MODEL WITH OPTIMIZED HYPER PARAMETERS TO TRAINING DATASET

training_slides = get_all_slices(CropTumor, holdout_filename_list)
# training_db = tf.data.Dataset.from_tensor_slices(training_slides) \
#     .map(preprocess) \
#     .batch(16) \
#     .prefetch(tf.data.AUTOTUNE) \
#     .shuffle(int(10e3))

# reset model weights
VAE.set_weights(initial_weights)

training_slides = training_slides.astype("float32") / 255.0

training_slides = np.reshape(training_slides,
                             newshape=(training_slides.shape[0], training_slides.shape[1], training_slides.shape[2], 1))
# fit model to the whole training database
training_results = VAE.fit(training_slides, training_slides,
                           epochs=10000,
                           callbacks=[early_stopping_training_db, tensorboard_callback],
                           verbose=1)

loss = VAE.evaluate(training_slides, training_slides,
                    verbose=1,
                    callbacks=[early_stopping_training_db, tensorboard_callback],
                    return_dict=False)

print("Model loss on hold out dataset is:")
print(loss)
