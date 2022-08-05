import tensorflow as tf
import tensorflow_probability as tfp
from Models.VAE_2.VAE_2_parameters import *

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfkb = tfk.backend


# function for sampling from mu and log_var
def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.random.normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1]), mean=0.0, stddev=1.0)
    random_sample = mu + tf.math.exp(log_variance / 2) * epsilon
    return random_sample


# Encoder
encoder_input_layer = tfkl.Input(shape=(206, 206, 1))
enc_conv_layer_1 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=1, activation='relu', name="conv_layer_1")(encoder_input_layer)
enc_batchnorm_layer_1 = tfkl.BatchNormalization(name="encoder_norm_1")(enc_conv_layer_1)
enc_conv_layer_2 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=2, activation='relu', name="conv_layer_2")(enc_batchnorm_layer_1)
enc_batchnorm_layer_2 = tfkl.BatchNormalization(name="encoder_norm_2")(enc_conv_layer_2)
enc_conv_layer_3 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=1, activation='relu', name="conv_layer_3")(enc_batchnorm_layer_2)
enc_batchnorm_layer_3 = tfkl.BatchNormalization(name="encoder_norm_3")(enc_conv_layer_3)
enc_conv_layer_4 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=2, activation='relu', name="conv_layer_4")(enc_batchnorm_layer_3)
enc_batchnorm_layer_4 = tfkl.BatchNormalization(name="encoder_norm_4")(enc_conv_layer_4)
enc_conv_layer_5 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=1, activation='relu', name="conv_layer_5")(enc_batchnorm_layer_4)
enc_batchnorm_layer_5 = tfkl.BatchNormalization(name="encoder_norm_5")(enc_conv_layer_5)
enc_conv_layer_6 = tfkl.Conv2D(filters=filters_number, kernel_size=3, strides=2, activation='relu', name="conv_layer_6")(enc_batchnorm_layer_5)
enc_batchnorm_layer_6 = tfkl.BatchNormalization(name="encoder_norm_6")(enc_conv_layer_6)
encoder_flatten_layer = tfkl.Flatten()(enc_batchnorm_layer_6)
encoder_mu_layer = tfkl.Dense(units=latent_dimensions, name="mu_encoder")(encoder_flatten_layer)
encoder_log_variance_layer = tfkl.Dense(units=latent_dimensions, name="log_var_encoder")(encoder_flatten_layer)
encoder_output_layer = tfkl.Lambda(sampling, name="encoder_output")([encoder_mu_layer, encoder_log_variance_layer])

# Build the encoder model
encoder = tf.keras.Model(encoder_input_layer, encoder_output_layer, name="encoder")

encoder.summary()

# Decoder
decoder_input_layer = tfkl.Input(shape=latent_dimensions)
dec_dense_layer = tfkl.Dense(units=23 * 23 * 64, activation=tf.nn.relu)(decoder_input_layer)
dec_reshape_layer = tfkl.Reshape(target_shape=(23, 23, 64))(dec_dense_layer)
dec_convT_layer_1 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=2, padding="valid", activation='relu', name="convT_layer_1")(dec_reshape_layer)
dec_batchnorm_layer_1 = tfkl.BatchNormalization(name="decoder_norm_1")(dec_convT_layer_1)
dec_convT_layer_2 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=1, padding='valid', activation='relu', name="convT_layer_2")(dec_batchnorm_layer_1)
dec_batchnorm_layer_2 = tfkl.BatchNormalization(name="decoder_norm_2")(dec_convT_layer_2)
dec_convT_layer_3 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=2, padding='valid', activation='relu', name="convT_layer_3")(dec_batchnorm_layer_2)
dec_batchnorm_layer_3 = tfkl.BatchNormalization(name="decoder_norm_3")(dec_convT_layer_3)
dec_convT_layer_4 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=1, padding='valid', activation='relu', name="convT_layer_4")(dec_batchnorm_layer_3)
dec_batchnorm_layer_4 = tfkl.BatchNormalization(name="decoder_norm_4")(dec_convT_layer_4)
dec_convT_layer_5 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=2, padding='same', activation='relu', name="convT_layer_5")(dec_batchnorm_layer_4)
dec_batchnorm_layer_5 = tfkl.BatchNormalization(name="decoder_norm_5")(dec_convT_layer_5)
dec_convT_layer_6 = tfkl.Conv2DTranspose(filters=filters_number, kernel_size=3, strides=1, padding='valid', activation='relu', name="convT_layer_6")(dec_batchnorm_layer_5)
dec_batchnorm_layer_6 = tfkl.BatchNormalization(name="decoder_norm_6")(dec_convT_layer_6)
dec_convT_layer_7 = tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='valid', activation='relu', name="convT_layer_7")(dec_batchnorm_layer_6)

# Build the decoder model
decoder = tf.keras.Model(decoder_input_layer, dec_convT_layer_7, name="decoder")
decoder.summary()


# Build VAE
VAE_input = tfkl.Input((206, 206, 1))
VAE_enc_output = encoder(VAE_input)

VAE = tfk.Model(VAE_input, decoder(VAE_enc_output))

VAE.summary()


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):

        reconstruction_loss = tf.math.reduce_sum(tf.math.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.math.reduce_sum(1.0 + encoder_log_variance - tf.math.square(encoder_mu) - tf.math.exp(encoder_log_variance),
                                  axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_weight*reconstruction_loss + kl_weight*kl_loss
        return loss

    return vae_loss


# Compile model
VAE.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_func(encoder_mu_layer, encoder_log_variance_layer))


early_stopping_kfold = tfk.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10,
                                                   verbose=2,
                                                   restore_best_weights=True)
early_stopping_training_db = tfk.callbacks.EarlyStopping(monitor="loss",
                                                         patience=10,
                                                         verbose=2,
                                                         restore_best_weights=True)
