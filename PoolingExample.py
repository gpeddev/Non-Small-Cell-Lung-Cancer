import tensorflow as tf
import os
from Models.VAE_1.VAE_1_model_1_parameters import batch_sz, latent_dimensions
from SupportCode.Paths import CropTumor
import numpy as np
vaes_latent_space=64

# Pooling features from each encoder trained in each fold (5)
from SupportCode.datasets_support import preprocess_data

model_directory = "./StoreResults/VAE_1/2022-08-03_02:06:12/Models"
encoder_1 = tf.keras.models.load_model(model_directory+'/VAE_encoder_1', compile=False)
encoder_2 = tf.keras.models.load_model(model_directory+'/VAE_encoder_2', compile=False)
encoder_3 = tf.keras.models.load_model(model_directory+'/VAE_encoder_3', compile=False)
encoder_4 = tf.keras.models.load_model(model_directory+'/VAE_encoder_4', compile=False)
encoder_5 = tf.keras.models.load_model(model_directory+'/VAE_encoder_5', compile=False)
encoders = [encoder_1, encoder_2, encoder_3, encoder_4, encoder_5]

results=[]

for enc in encoders:
    patient_dict = {}
    for patient in os.listdir(CropTumor):
        # load all slices of patient
        patient_slices = preprocess_data(CropTumor, [patient])
        # apply encoder to each slice

        patient_feature_array = np.empty((0, vaes_latent_space))
        for patient_sl in range(patient_slices.shape[0]):
            temp = patient_slices[patient_sl, :, :]
            temp_reshaped = np.reshape(temp, newshape=(1, temp.shape[0], temp.shape[1], 1))
            enc_result = enc(temp_reshaped)[0].numpy()
            enc_result = np.reshape(enc_result,newshape=(1,vaes_latent_space))
            patient_feature_array = np.vstack([patient_feature_array, enc_result])
        # store outcome to dictionary {patient_name : feature numpy array [number of features, length to latent space]}
        average_pc=np.average(patient_feature_array, axis=0)
        max_pc=np.max(patient_feature_array, axis=0)
        patient_dict[patient] = np.concatenate([average_pc, max_pc])
    results.append(patient_dict)

results

    # apply average per column
    # apply max per column
    # concatenate to feature
    # add to dictionary a pair {patient: extracted feature}
