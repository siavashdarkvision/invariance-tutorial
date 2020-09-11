import os

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from src.plot_tools import pics_tools as pic

NUM_LABELS = 10
IMG_DIM = 28
MAX_INTENSITY = 255.0
DIM_Z = 16
DIM_C = NUM_LABELS
INPUT_SHAPE = IMG_DIM ** 2
ACTIVATION = "tanh"


def all_pairs_gaussian_kl(mu, sigma, add_third_term=False):
    """KL(N_0|N_1) = tr(\sigma_1^{-1} \sigma_0) +
        (\mu_1 - \mu_0)\sigma_1^{-1}(\mu_1 - \mu_0) - k +
        \log( \frac{\det \sigma_1}{\det \sigma_0} )

    """
    sigma_sq = tf.square(sigma) + 1e-8
    sigma_sq_inv = tf.math.reciprocal(sigma_sq)

    # The dot product of all sigma_inv vectors with sigma the same as a matrix
    # multiplication of diagionals.
    first_term = tf.matmul(sigma_sq,tf.transpose(sigma_sq_inv))

    r = tf.matmul(mu * mu,tf.transpose(sigma_sq_inv))
    r2 = mu * mu * sigma_sq_inv
    r2 = tf.reduce_sum(r2,1)

    # Squared distance:
    # (mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
    # uses broadcasting
    second_term = 2*tf.matmul(mu, tf.transpose(mu*sigma_sq_inv))
    second_term = r - second_term + tf.transpose(r2)

    # log det A = tr log A
    # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
    #   \tr\log \Sigma_1 - \tr\log \Sigma_0
    #
    # for each sample, we have B comparisons to B other samples...
    #   so this cancels out
    if(add_third_term):
        r = tf.reduce_sum(tf.math.log(sigma_sq),1)
        r = tf.reshape(r,[-1,1])
        third_term = r - tf.transpose(r)
    else:
        third_term = 0

    #- tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)))\
    # the dim_z ** 3 term comes from:
    #   -the k in the original expression
    #   -this happening k times in for each sample
    #   -this happening for k samples
    return 0.5 * ( first_term + second_term + third_term )


def kl_conditional_and_marg(z_mean, z_log_sigma_sq, dim_z):
    """\sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
    """
    z_sigma = tf.exp( 0.5 * z_log_sigma_sq )
    all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5*dim_z
    return tf.reduce_mean(all_pairs_GKL)


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def one_hot(labels):
    num_labels_data = labels.shape[0]
    one_hot_encoding = np.zeros((num_labels_data, NUM_LABELS))
    one_hot_encoding[np.arange(num_labels_data), labels] = 1
    one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


def get_mnist():
    #data comes as images and 1-dim {0,...,9} categorical variable
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    #cast and flatten images, renormalizing to [0,1]
    train_x = train_x.astype(np.float32).reshape( (train_x.shape[0], IMG_DIM ** 2) ) / MAX_INTENSITY
    test_x = test_x.astype(np.float32).reshape( (test_x.shape[0], IMG_DIM ** 2) ) / MAX_INTENSITY

    train_y = one_hot(train_y).astype(np.float32)
    test_y = one_hot(test_y).astype(np.float32)

    return (train_x, train_y), (test_x, test_y)


def encoder(input_x):
    enc_hidden_1 = keras.layers.Dense(512, activation=ACTIVATION, name="enc_h1")(input_x)
    enc_hidden_2 = keras.layers.Dense(512, activation=ACTIVATION, name="enc_h2")(enc_hidden_1)

    z_mean = keras.layers.Dense(DIM_Z, activation=ACTIVATION)(enc_hidden_2)
    z_log_sigma_sq = keras.layers.Dense(DIM_Z, activation="linear")(enc_hidden_2)

    z = keras.layers.Lambda(sampling, output_shape=(DIM_Z,), name='z')([z_mean, z_log_sigma_sq])

    return z, z_mean, z_log_sigma_sq


def decoder(z, input_c):
    z_with_c = keras.layers.concatenate([z,input_c])

    dec_hidden_1 = keras.layers.Dense(512, activation=ACTIVATION, name="dec_h1")(z_with_c)
    dec_hidden_2 = keras.layers.Dense(512, activation=ACTIVATION, name="dec_h2")(dec_hidden_1)

    x_hat = keras.layers.Dense( INPUT_SHAPE, name="x_hat", activation="linear" )(dec_hidden_2)
    return x_hat


def get_loss(input_x, x_hat, z_mean, z_log_sigma_sq, params):
    recon_loss = keras.losses.mse(input_x, x_hat)
    recon_loss *= INPUT_SHAPE

    prior_loss = 1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq)
    prior_loss = K.sum(prior_loss, axis=-1)
    prior_loss *= -0.5

    kl_qzx_qz_loss = kl_conditional_and_marg(z_mean, z_log_sigma_sq, DIM_Z)

    loss = K.mean((1 + params["lambda"]) * recon_loss +
        params["beta"] * prior_loss + params["lambda"] * kl_qzx_qz_loss)

    return loss


def get_model(params):
    input_x = keras.layers.Input(shape = [INPUT_SHAPE], name="x")
    input_c = keras.layers.Input(shape = [DIM_C], name="c")

    z, z_mean, z_log_sigma_sq = encoder(input_x)
    x_hat = decoder(z, input_c)

    model = keras.models.Model(inputs=[input_x, input_c], outputs=x_hat, name="ICVAE")
    loss = get_loss(input_x, x_hat, z_mean, z_log_sigma_sq, params)
    model.add_loss(loss)

    return model


def visualize_samples(model, test_x, test_y, n_samples=10):
    X_test_set, Y_test_set = [], []

    for i in range(n_samples):
        tmp_tile_array = np.tile(test_x[i],[10, 1])
        X_test_set.append(test_x[i:(i+1),:])
        X_test_set.append(tmp_tile_array)

        Y_test_set.append(test_y[i:(i+1),:])
        Y_test_set.append(np.eye(10))

    X_test_set = np.concatenate(X_test_set, axis=0)
    Y_test_set = np.concatenate(Y_test_set, axis=0)

    X_test_hat = model.predict({ "x" : X_test_set, "c" : Y_test_set })

    plot_collection = []
    for i in range(n_samples):
        plot_collection.append( test_x[i:(i+1),:] )
        plot_collection.append( X_test_hat[i*11:(i+1)*11,:] )

    plot_collection = np.concatenate(plot_collection, axis=0)

    fig = pic.plot_image_grid(1-plot_collection, [IMG_DIM, IMG_DIM], (n_samples, 12))
    plt.show()


def main():
    (train_x, train_y), (test_x, test_y) = get_mnist()
    params = {"beta" : 0.1, "lambda" : 1.0, "learning_rate": 0.0005}
    model = get_model(params)

    opt = keras.optimizers.Adam(lr=params["learning_rate"])
    model.compile(optimizer=opt)

    if not os.path.exists("mnist_icvae.h5"):
        print("training")
        model.fit({ "x" : train_x, "c" : train_y }, epochs=100)
        model.save_weights("mnist_icvae.h5")
    else:
        print("loading from file")
        model.load_weights("mnist_icvae.h5")

    visualize_samples(model, test_x, test_y)


if __name__ == '__main__':
    main()
