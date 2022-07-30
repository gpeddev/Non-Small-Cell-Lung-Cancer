########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import os
from datetime import datetime
import random
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from SupportCode.datasets_support import preprocess_data
from Models.VAE_1.VAE_1_parameters import *
from Models.VAE_1.VAE_model_1_6LAYERS import VAE, early_stopping_kfold, tfk, encoder, decoder
from SupportCode.Paths import CropTumor


# Store initial model weights for resets
initial_weights = VAE.get_weights()


# Get data from the appropriate directory
file_array = os.listdir(CropTumor)
file_array = np.array(file_array)           # array of patients filenames

kFold = KFold(n_splits=5, shuffle=True, random_state=1)

counter = 1     # counter used for numbering the stored models
kFold_results = []      # hold history output of fitting the model

time_started = datetime.now()

# k-fold initial splits of our datasets to 5 test and converge datasets
for converge_dataset, test_dataset in kFold.split(file_array):          # kfold split gives indexes sets

    # tensorboard
    log_dir = "./Output/Logs/model_1_tunning" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')

    # Secondary split of converge dataset to training and valuation dataset (train and val datasets are indexes)
    train_dataset, val_dataset = train_test_split(converge_dataset, test_size=0.20, shuffle=True)

    # get database ready for our models. shape => (slice number, slice width, slice height, 1(slice depth) )
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

    # store trained models
    VAE.save("./Output/Models/VAE_" + str(counter))
    encoder.save("./Output/Models/VAE_encoder_" + str(counter))
    decoder.save("./Output/Models/VAE_decoder_" + str(counter))

    counter = counter + 1

    kFold_results.append(fit_results)

time_ended = datetime.now()

# Calculate average validation loss across the k models.
add_val_losses = 0
for item in kFold_results:
    add_val_losses = add_val_losses + item.history["val_loss"][-1]

print("Average Loss at kfold at valuation data is: ")
print(add_val_losses / len(kFold_results))
print("Time started: ", time_started)
print("Time started: ", time_ended)
print("learning rate:", learning_rate)
print("latent_dimentions:", latent_dimensions)
print("filters_number:", filters_number)
print("kernels_number:", kernels_number)
print("stride:", stride)
print("kl_weight:", kl_weight)
print("reconstruction_weight:", reconstruction_weight)


# Notify finish training
duration = 1  # seconds
freq = 440  # Hz
os.system('spd-say "Your program has finished"')
