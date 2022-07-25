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
latent_dimentions = 64
filters_number = 32
kernels_number = 3
stride = 2
kl_weight = 3

# Prior
def sampling(mu_log_variance):
    mu, log_variance= tf.split(mu_log_variance, num_or_size_splits=2, axis=1)
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

# Encoder

encoder = tfk.Sequential(
    [
        tfkl.InputLayer(input_shape=(138, 138, 1)),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(2,2), strides=2, activation='relu'),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=2, strides=1, activation='relu'),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=2, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=2, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=2, strides=1, activation='relu'),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=2, strides=2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2 * latent_dimentions,name="mu_and_logvar"),
        tf.keras.layers.Lambda(sampling),
    ], name="Encoder"
)
encoder.summary()

decoder = tfk.Sequential(
    [
        tfkl.InputLayer(input_shape=(latent_dimentions, )),
        tfkl.Dense(units=8*8*128, activation=tf.nn.relu),
        tfkl.Reshape(target_shape=(8, 8, 128)),
        tfkl.Conv2DTranspose(
                filters=32, kernel_size=2, strides=2, padding="same",
                activation='relu'),
        tfkl.Conv2DTranspose(
            filters=32, kernel_size=2, strides=1, padding="valid",
            activation='relu'),
        tfkl.Conv2DTranspose(
            filters=64, kernel_size=2, strides=2, padding='same',
            activation='relu'),
        tfkl.Conv2DTranspose(
            filters=64, kernel_size=2, strides=2, padding='same',
            activation='relu'),
        tfkl.Conv2DTranspose(
            filters=128, kernel_size=2, strides=1, padding='valid',
            activation='relu'),
        tfkl.Conv2DTranspose(
            filters=128, kernel_size=2, strides=2, padding='same',
            activation='relu'),
        tfkl.Conv2DTranspose(
            filters=1, kernel_size=2, strides=1, padding='same',
            activation='relu'),
    ], name="Decoder"
)
decoder.summary()

VAE = tfk.models.Model(encoder.inputs, decoder(encoder.outputs), name="VAE")



def loss_func(y_true, y_predict):
    mu, log_variance= tf.split(encoder["mu_and_logvar"].output, num_or_size_splits=2, axis=1)

    kl_loss = -0.5 * tfk.backend.sum(
        1.0 + log_variance - tfk.backend.square(mu) - tfk.backend.exp(log_variance), axis=1)

    reconstruction_loss = tfk.backend.mean(tfk.backend.square(y_true - y_predict), axis=[1, 2, 3])
    loss = reconstruction_loss + kl_loss
    return loss



VAE.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), loss=loss_func)




early_stopping_kfold = tfk.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=50,
                                                   verbose=2)

early_stopping_training_db = tfk.callbacks.EarlyStopping(monitor="loss",
                                                         patience=50,
                                                         verbose=2,
                                                         restore_best_weights=True)

VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))

VAE.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError())
