########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import os
from datetime import datetime
import random

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from Datasets_splits.splited_datasets import preprocess_data
from Models.VAE_1.VAE_1_parameters import batch_size, learning_rate, kernels_number, filters_number, latent_dimensions
from Models.VAE_1.VAE_model_1 import VAE, early_stopping_kfold, tfk
from Paths import CropTumor

# Store initial model weights for resets
initial_weights = VAE.get_weights()


# ############################################################################################### KFOLD CROSS VALIDATION
file_array = os.listdir(CropTumor)
random.shuffle(file_array)
file_array = np.array(file_array)

# init_train is the part used in kfold cross validation
# init_test is the part used for testing

kFold = KFold(n_splits=5, shuffle=True, random_state=1)
kFold_results = []
time_started = datetime.now()

for converge_dataset, test_dataset in kFold.split(file_array):

    log_dir = "./Logs/model_1_hyperparameter_tunning" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
    train_dataset, val_dataset = train_test_split(converge_dataset, test_size=0.10, shuffle=True)

    train_slices = preprocess_data(CropTumor, file_array[train_dataset])
    val_slices = preprocess_data(CropTumor, file_array[val_dataset])

    # reset model weights before training
    VAE.set_weights(initial_weights)

    # fit model
    fit_results = VAE.fit(train_slices, train_slices,
                          epochs=100000,
                          validation_data=(val_slices, val_slices),
                          batch_size=batch_size,
                          shuffle=True,
                          callbacks=[early_stopping_kfold, tensorboard_callback],
                          verbose=2)
    kFold_results.append(fit_results)
time_ended = datetime.now()

# ######################################################################################## AVERAGE VALIDATION FROM KFOLD
add_val_losses = 0
for item in kFold_results:
    add_val_losses = add_val_losses + item.history["val_loss"][-1]

print("Average Loss at kfold at valuation data is: ")
print(add_val_losses / len(kFold_results))
print("Time started: ", time_started)
print("Time started: ", time_ended)
print("learning rate:",learning_rate)
print("latent_dimentions:",latent_dimensions)
print("filters_number:",filters_number)
print("kernels_number:",kernels_number)
print("stride:",stride)

# from matplotlib import pyplot as plt
# data=VAE.predict(whole_training_dataset[10:15])
# plt.imshow(np.reshape(data[0], newshape=(138, 138))*255, interpolation='nearest')
# plt.show()
# plt.imshow(np.reshape(whole_training_dataset[10:15][0], newshape=(138, 138)) * 255)
# plt.show()


# ###################################################### TRAIN MODEL WITH OPTIMIZED HYPER PARAMETERS TO TRAINING DATASET
# # reset model weights
# VAE.set_weights(initial_weights)
#
# # # slides from the hold out dataset
# # whole_training_dataset = get_all_slices(CropTumor, holdout_filename_list)
# #
# # # adjust to 0 - 1 range and shuffle CTs
# # whole_training_dataset = whole_training_dataset.astype("float32") / 255
# #
# # whole_training_dataset = np.reshape(whole_training_dataset,
# #                                     newshape=(whole_training_dataset.shape[0], whole_training_dataset.shape[1], whole_training_dataset.shape[2], 1))
# whole_training_dataset = preprocess_data(CropTumor,holdout_filename_list)
# # fit model to the whole training database
# training_results = VAE.fit(whole_training_dataset, whole_training_dataset,
#                            epochs=10000,
#                            batch_size=batch_size,
#                            callbacks=[early_stopping_training_db, tensorboard_callback],
#                            verbose=1)
#
# loss = VAE.evaluate(whole_training_dataset, whole_training_dataset,
#                     verbose=1,
#                     callbacks=[early_stopping_training_db, tensorboard_callback],
#                     return_dict=False)
#
# print("Model loss on hold out dataset is:")
# print(loss)
#
# ######################################################################## TRAIN MODEL TO THE WHOLE DATASET FOR PREDICTION
# # reset model
# VAE.set_weights(initial_weights)
# #
# # print("Train model to the whole dataset:")
# #
# # # Must train to the whole dataset now to make predictions
# # whole_dataset = get_all_slices(CropTumor, file_array)
# #
# # # adjust to 0 - 1 range and shuffle CTs
# # whole_dataset = whole_dataset.astype("float32") / 255
# #
# # whole_dataset = np.reshape(whole_dataset,
# #                              newshape=(whole_dataset.shape[0], whole_dataset.shape[1], whole_dataset.shape[2], 1))
#
# whole_dataset = preprocess_data(CropTumor, file_array)
#
# # fit model to the whole training database
# whole_results = VAE.fit(whole_dataset, whole_dataset,
#                            epochs=10000,
#                            batch_size=batch_size,
#                            callbacks=[early_stopping_training_db, tensorboard_callback],
#                            verbose=1)
#
#
# from matplotlib import pyplot as plt
# data=VAE.predict(whole_training_dataset[10:15])
# plt.imshow(np.reshape(data[0], newshape=(138, 138))*255, interpolation='nearest')
# plt.show()
# plt.imshow(np.reshape(whole_training_dataset[10:15][0], newshape=(138, 138)) * 255)
# plt.show()
