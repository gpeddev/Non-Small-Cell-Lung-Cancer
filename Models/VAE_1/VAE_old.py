import tensorflow as tf
import tensorflow_probability as tfp
import datetime

# Shortcuts
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

log_dir = "./Logs/mse" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir,update_freq='epoch')

################################################### Mean Square Error ##################################################

# Preprocess images
def preprocess(sample):
    image = tf.cast(sample, tf.float64) / 255.  # Scale to unit interval.
    return image, image


# Training options
learning_rate = 1e-4
latent_dimentions = 128
filters_number = 32
kernels_number = 3
stride = 2
kl_weight = 1

# Prior
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dimentions), scale=1),
                        reinterpreted_batch_ndims=1)

# Encoder
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=(138, 138, 1)),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=stride,
                activation="relu"),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=stride,
                activation="relu"),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=stride,
                activation="relu"),
    tfkl.Flatten(),
    tfkl.Dense(units=tfpl.MultivariateNormalTriL.params_size(latent_dimentions),
               activation=None),
    tfpl.MultivariateNormalTriL(event_size=latent_dimentions,
                                convert_to_tensor_fn=tfd.Distribution.sample,
                                activity_regularizer=tfpl.KLDivergenceRegularizer(prior,
                                                                                  weight=kl_weight)),
])
encoder.summary()

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=latent_dimentions),
    tfkl.Dense(16*16*latent_dimentions, activation=None),
    tfkl.Reshape((16, 16, latent_dimentions)),
    tfkl.Conv2DTranspose(filters=filters_number,
                         kernel_size=3,
                         strides=2,
                         activation="relu"),
    tfkl.Conv2DTranspose(filters=filters_number,
                         kernel_size=4,
                         strides=2,
                         activation="relu"),
    tfkl.Conv2DTranspose(filters=filters_number,
                         kernel_size=4,
                         strides=2,
                         activation="relu"),
    tfkl.Conv2DTranspose(filters=1,
                         kernel_size=3,
                         strides=1,
                         padding="same",
                         activation=None),
])
decoder.summary()


vae = tfk.Model(inputs=encoder.input, outputs=decoder(encoder.output))
vae.compile(loss=lambda x, pred: -pred.log_prob(x))


early_stopping_kfold = tfk.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=30,
                                                   verbose=1)

early_stopping_training_db = tfk.callbacks.EarlyStopping(monitor="loss",
                                                         patience=30,
                                                         verbose=1,
                                                         restore_best_weights=True)

VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

VAE.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError())





########################################################################################################################
#                                                       VAE 1                                                          #
########################################################################################################################
import SimpleITK as Sitk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
from Paths import CropTumor
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt


# Helpers functions
# Returns an array with all the slices from the file_list CTs
def get_all_slices(filepath, file_list):
    results = []
    for filename in file_list:
        image_data = Sitk.GetArrayFromImage(Sitk.ReadImage(filepath+filename, Sitk.sitkUInt8))
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

# ############################################################################################### KFOLD CROSS VALIDATION

kFold = KFold(n_splits=5, shuffle=True, random_state=1)
kFold_results = []

for train, val in kFold.split(train_filename_list):
    # build train and valuation datasets
    train_slices = get_all_slices(CropTumor, train_filename_list[train])
    training_db = tf.data.Dataset.from_tensor_slices(train_slices)\
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
    fit_results = VAE.fit(training_db,
                          epochs=1000,
                          validation_data=val_db,
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
training_db = tf.data.Dataset.from_tensor_slices(training_slides) \
    .map(preprocess) \
    .batch(16) \
    .prefetch(tf.data.AUTOTUNE) \
    .shuffle(int(10e3))

# reset model weights
VAE.set_weights(initial_weights)
# fit model to the whole training database
training_results = VAE.fit(training_db,
                           epochs=10000,
                           callbacks=[early_stopping_training_db, tensorboard_callback],
                           verbose=1)

loss = VAE.evaluate(training_db,
                    verbose=1,
                    callbacks=[early_stopping_training_db, tensorboard_callback],
                    return_dict=False)

print("Model loss on hold out dataset is:")
print(loss)
