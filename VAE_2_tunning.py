########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import glob
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from SupportCode.datasets_support import preprocess_data, create_image_augmentation_dir
from Models.VAE_2.VAE_2_parameters import *
from Models.VAE_2.VAE_2_model_1_6_layers_a1 import VAE, early_stopping_kfold, tfk, encoder, decoder
from SupportCode.Paths import CroppedWindow
import tensorflow as tf
# Store initial model weights for resets
initial_weights = VAE.get_weights()


def preprocess_dataset_train(image):
    image = tf.cast(image, tf.float32) / 255  # Scale to unit interval.
    return image, image


def preprocess_dataset_valuation(image):
    return image, image


# Get data from the appropriate directory
file_array = os.listdir(CroppedWindow)
file_array = np.array(file_array)           # array of patients filenames

kFold = KFold(n_splits=5, shuffle=True, random_state=1)

counter = 1     # counter used for numbering the stored models
kFold_results = []      # hold history output of fitting the model

time_started = datetime.now()

# k-fold initial splits of our datasets to 5 test and converge datasets
for converge_dataset, test_dataset in kFold.split(file_array):          # kfold split gives indexes sets

    # Secondary split of converge dataset to training and valuation dataset (train and val datasets are indexes)
    train_dataset, val_dataset = train_test_split(converge_dataset, test_size=0.20, shuffle=True, random_state=1)

    # tensorboard
    log_dir = "./Output/Logs/model_1_tuning" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')

    # get database ready for our models. shape => (slice number, slice width, slice height, 1(slice depth) )
    train_slices = preprocess_data(CroppedWindow, file_array[train_dataset])
    val_slices = preprocess_data(CroppedWindow, file_array[val_dataset])

    # training dataset
    # create directory with augmented images
    create_image_augmentation_dir(train_slices, growth_factor=10, save_to_path="./Data/10_TrainingSet_VAE2")
    # create a dataset from directory
    train_dset = tf.keras.preprocessing.image_dataset_from_directory(directory="./Data/10_TrainingSet_VAE2",
                                                                     labels=None,
                                                                     label_mode=None,
                                                                     image_size=(206, 206),
                                                                     color_mode="grayscale",
                                                                     batch_size=None,
                                                                     shuffle=True)

    train_dset = (train_dset.map(preprocess_dataset_train)
                  .cache()
                  .batch(batch_sz)
                  .prefetch(tf.data.AUTOTUNE)
                  .shuffle(1))

    # validation dataset
    val_dset = tf.data.Dataset.from_tensor_slices(val_slices)
    val_dset = (val_dset.map(preprocess_dataset_valuation)
                .cache()
                .batch(batch_sz)
                .prefetch(tf.data.AUTOTUNE)
                .shuffle(1))

    # reset model weights before training
    VAE.set_weights(initial_weights)

    # fit model
    fit_results = VAE.fit(train_dset,
                          epochs=2,
                          validation_data=val_dset,
                          callbacks=[early_stopping_kfold, tensorboard_callback],
                          verbose=2
                          )

    # clean dataset directory
    for f in glob.glob("./Data/10_TrainingSet_VAE2/*.png"):
        os.remove(f)

    # store trained models
    VAE.save("./Output/VAE_2/Models/VAE_" + str(counter)+".h5")
    encoder.save("./Output/VAE_2/Models/VAE_encoder_" + str(counter)+".h5")
    decoder.save("./Output/VAE_2/Models/VAE_decoder_" + str(counter)+".h5")

    # store filenames for each dataset
    np.save("./Output/VAE_2/DatasetSplits/" + "test_dataset_fold_" + str(counter), file_array[test_dataset])
    np.save("./Output/VAE_2/DatasetSplits/" + "val_dataset_fold_" + str(counter), file_array[val_dataset])
    np.save("./Output/VAE_2/DatasetSplits/" + "train_dataset_fold_" + str(counter), file_array[train_dataset])

    counter = counter + 1

    kFold_results.append(fit_results)

time_ended = datetime.now()

# Calculate average validation loss across the k models.
add_val_losses = 0
for item in kFold_results:
    add_val_losses = add_val_losses + item.history["val_loss"][-1]

output = "Average Loss at kfold at valuation data is: " + str(add_val_losses / len(kFold_results)) + "\n" + \
         "Time started: " + str(time_started) + "\n" + \
         "Time ended: " + str(time_ended) + "\n" + \
         "learning rate: " + str(learning_rate) + "\n" + \
         "latent_dimensions: " + str(latent_dimensions) + "\n" +\
         "filters_number: " + str(filters_number) + "\n" +\
         "batch_size: " + str(batch_sz) + "\n" + \
         "kl_weight: " + str(kl_weight) + "\n" + \
         "reconstruction_weight: " + str(reconstruction_weight)

print(output)

# save hyperparameters to file
text_file = open("./Output/VAE_2/hyperparameters.txt", "w")
text_file.write(output)
text_file.close()

# Notify finish training
os.system('spd-say "Training finished!"')
