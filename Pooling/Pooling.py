import tensorflow as tf
import os
from Models.VAE_1.VAE_1_model_1_parameters import batch_sz, latent_dimensions
from SupportCode.Paths import CropTumor
import numpy as np

# Pooling features from each encoder trained in each fold (5)
from SupportCode.datasets_support import preprocess_data



def pooling(encoder_path, vaes_latent_space, source_images):

    results = []
    model_directory = encoder_path
    encoder_1 = tf.keras.models.load_model(model_directory + '/VAE_encoder_1.h5', compile=False)
    encoder_2 = tf.keras.models.load_model(model_directory + '/VAE_encoder_2.h5', compile=False)
    encoder_3 = tf.keras.models.load_model(model_directory + '/VAE_encoder_3.h5', compile=False)
    encoder_4 = tf.keras.models.load_model(model_directory + '/VAE_encoder_4.h5', compile=False)
    encoder_5 = tf.keras.models.load_model(model_directory + '/VAE_encoder_5.h5', compile=False)
    encoders = [encoder_1, encoder_2, encoder_3, encoder_4, encoder_5]

    i = 1
    for enc in encoders:
        print("Pooling on encoder "+str(i))
        patient_dict = {}
        for patient in os.listdir(source_images):
            # load all slices of patient
            patient_slices = preprocess_data(source_images, [patient])
            # apply encoder to each slice

            patient_feature_array = np.empty((0, vaes_latent_space))
            for patient_sl in range(patient_slices.shape[0]):
                temp = patient_slices[patient_sl, :, :]
                temp_reshaped = np.reshape(temp, newshape=(1, temp.shape[0], temp.shape[1], 1))
                enc_result = enc(temp_reshaped)[0].numpy()
                enc_result = np.reshape(enc_result, newshape=(1, vaes_latent_space))
                patient_feature_array = np.vstack([patient_feature_array, enc_result])
            # store outcome to dictionary {patient_name : feature numpy array [number of features, length to latent space]}
            average_pc = np.average(patient_feature_array, axis=0)
            max_pc = np.max(patient_feature_array, axis=0)
            patient_dict[patient.rsplit(".")[0]] = np.concatenate([average_pc, max_pc])
        results.append(patient_dict)
        i=i+1
    print("Pooling finished")
    return results

#rslt=pooling("./BestResults/VAE_1/Model_1/19_2022-08-10_01_53_19/Models", 256,CropTumor)
