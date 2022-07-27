from datetime import datetime

from Datasets_splits.splited_datasets import preprocess_data, holdout_filename_list
from Models.VAE_1.VAE_1_parameters import *
from Models.VAE_1.VAE_model_1 import VAE, early_stopping_training_db
from Paths import CropTumor
from tensorflow import keras as tfk


log_dir = "./Logs/model_1_holdout_score" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tfk.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')

whole_training_dataset = preprocess_data(CropTumor,holdout_filename_list)
# fit model to the whole training database
training_results = VAE.fit(whole_training_dataset, whole_training_dataset,
                           epochs=10000,
                           batch_size=batch_size,
                           callbacks=[early_stopping_training_db, tensorboard_callback],
                           verbose=1)

loss = VAE.evaluate(whole_training_dataset, whole_training_dataset,
                    verbose=1,
                    callbacks=[early_stopping_training_db, tensorboard_callback],
                    return_dict=False)

print("Model loss on hold out dataset is:")
print(loss)