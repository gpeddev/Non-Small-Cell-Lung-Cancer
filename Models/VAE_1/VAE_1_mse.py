
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
latent_dimentions = 256
filters_number = 32
kl_weight = 1

# Prior
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dimentions), scale=1),
                        reinterpreted_batch_ndims=1)

# Encoder
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=(138, 138, 1)),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=2,
                activation="relu"),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=2,
                activation="relu"),
    tfkl.Conv2D(filters=filters_number,
                kernel_size=3,
                strides=2,
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

early_stopping_kfold = tfk.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=50,
                                                   verbose=1)

early_stopping_training_db = tfk.callbacks.EarlyStopping(monitor="loss",
                                                         patience=50,
                                                         verbose=1,
                                                         restore_best_weights=True)

VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

VAE.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.MeanSquaredError())
