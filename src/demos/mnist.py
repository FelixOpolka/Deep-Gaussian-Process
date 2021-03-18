import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import time

from gpflow.kernels import RBF
from gpflow.likelihoods import MultiClass
from scipy.cluster.vq import kmeans2
from scipy.stats import mode

from deep_gp import DeepGP


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train = (x_train.astype(np.float64) / 255.0) - 0.5
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = (x_test.astype(np.float64) / 255.0) - 0.5
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, y_train, x_test, y_test


def make_dgp(num_layers, X, Y, Z):
    kernels = [RBF(variance=2.0, lengthscales=2.0)]
    layer_sizes = [784]
    for l in range(num_layers-1):
        kernels.append(RBF(variance=2.0, lengthscales=2.0))
        layer_sizes.append(30)
    model = DeepGP(X, Y, Z, kernels, layer_sizes, MultiClass(10),
                   num_outputs=10)

    # init hidden layers to be near deterministic
    for layer in model.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    return model


def training_step(model, X, Y, batch_size=1000):
    n_batches = max(int(len(x_train) / batch_size), 1)
    elbos = []
    for x_batch, y_batch in zip(np.split(X, n_batches),
                                np.split(Y, n_batches)):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(dgp.trainable_variables)
            objective = -model.elbo((x_batch, y_batch))
            gradients = tape.gradient(objective, dgp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dgp.trainable_variables))
        elbos.append(-objective.numpy())
    return np.mean(elbos)


def evaluation_step(model, X, Y, batch_size=1000, num_samples=100):
    n_batches = max(int(len(X) / batch_size), 1)
    likelihoods, accs = [], []
    for x_batch, y_batch in zip(np.split(X, n_batches),
                                np.split(Y, n_batches)):
        m, v = model.predict_y(x_batch, num_samples)
        likelihood = model.predict_log_density((x_batch, y_batch), num_samples)
        acc = (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
        likelihoods.append(likelihood)
        accs.append(acc)
    return np.mean(likelihoods), np.mean(accs)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    num_inducing = 100
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    batch_size = 1000
    num_samples = 100

    dgp = make_dgp(2, x_train, y_train, Z)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    for _ in range(1500):
        start_time = time.time()
        elbo = training_step(dgp, x_train, y_train, batch_size)
        likelihood, acc = evaluation_step(dgp, x_test, y_test, batch_size,
                                          num_samples)
        duration = time.time() - start_time
        print(f"ELBO: {elbo}, Likelihood: {likelihood}, Acc: {acc} [{duration}]")
