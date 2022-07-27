from datetime import datetime

from Datasets_splits.splited_datasets import preprocess_data, file_array
from Models.VAE_1.VAE_1_parameters import *
from Models.VAE_1.VAE_model_1 import VAE, early_stopping_training_db
from Paths import CropTumor
from tensorflow import keras as tfk


# Callbacks definition. tensorboard and earlystopping
log_dir = "./Logs/model_1_holdout_score" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')

# early_stopping_training_db = tfk.callbacks.EarlyStopping(monitor="loss",
#                                                          patience=30,
#                                                          verbose=2,
#                                                          restore_best_weights=True)


whole_dataset = preprocess_data(CropTumor, file_array)

# fit model to the whole training database
whole_results = VAE.fit(whole_dataset, whole_dataset,
                           epochs=10000,
                           batch_size=batch_size,
                           callbacks=[early_stopping_training_db, tensorboard_callback],
                           verbose=1)

VAE.save('./SavedModels/model_1.model')


# from matplotlib import pyplot as plt
# data=VAE.predict(whole_training_dataset[10:15])
# plt.imshow(np.reshape(data[0], newshape=(138, 138))*255, interpolation='nearest')
# plt.show()
# plt.imshow(np.reshape(whole_training_dataset[10:15][0], newshape=(138, 138)) * 255)
# plt.show()
