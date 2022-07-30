import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

# load models
from SupportCode.datasets_support import preprocess_data
from SupportCode.Paths import CropTumor

encoder_1 = tf.keras.models.load_model('./SavedModels/VAE_encoder_1',compile=False)
encoder_2 = tf.keras.models.load_model('./SavedModels/VAE_encoder_2',compile=False)
encoder_3 = tf.keras.models.load_model('./SavedModels/VAE_encoder_3',compile=False)
encoder_4 = tf.keras.models.load_model('./SavedModels/VAE_encoder_4',compile=False)
encoder_5 = tf.keras.models.load_model('./SavedModels/VAE_encoder_5',compile=False)

decoder_1 = tf.keras.models.load_model('./SavedModels/VAE_decoder_1',compile=False)
decoder_2 = tf.keras.models.load_model('./SavedModels/VAE_decoder_2',compile=False)
decoder_3 = tf.keras.models.load_model('./SavedModels/VAE_decoder_3',compile=False)
decoder_4 = tf.keras.models.load_model('./SavedModels/VAE_decoder_4',compile=False)
decoder_5 = tf.keras.models.load_model('./SavedModels/VAE_decoder_5',compile=False)

vae_1 = tf.keras.models.load_model('./SavedModels/VAE_1',compile=False)
vae_2 = tf.keras.models.load_model('./SavedModels/VAE_2',compile=False)
vae_3 = tf.keras.models.load_model('./SavedModels/VAE_3',compile=False)
vae_4 = tf.keras.models.load_model('./SavedModels/VAE_4',compile=False)
vae_5 = tf.keras.models.load_model('./SavedModels/VAE_5',compile=False)

# vae_1 = tf.keras.models.load_model('./Results/2022-07-29_09:19:59/SavedModels/VAE_1',compile=False)

file_array = os.listdir(CropTumor)
data=preprocess_data(CropTumor,file_array)

# iterate data and get features
for check_image in range(50):

    temp = np.reshape(data[check_image], newshape=(1, data[check_image].shape[0], data[check_image].shape[1], 1))
    result = vae_1(temp)
    before = np.reshape(data[check_image], newshape=(data[check_image].shape[0], data[check_image].shape[1]))
    plt.imshow(before, interpolation='nearest')
    plt.show()
    plt.imsave("./Images/"+str(check_image) + 'a.jpeg',before)
    after = np.reshape(result[0], newshape=(result[0].shape[0], result[0].shape[1]))
    plt.imshow(after, interpolation='nearest')
    plt.show()
    plt.imsave("./Images/"+str(check_image) + 'b.jpeg', after)

#
# temp = np.reshape(data[2],newshape=(1,data[2].shape[0],data[2].shape[1],1))
# result = vae_1(temp)
#
# plt.imshow(data[2], interpolation='nearest')
# plt.show()
# plt.imshow(result[0], interpolation='nearest')
# plt.show()
