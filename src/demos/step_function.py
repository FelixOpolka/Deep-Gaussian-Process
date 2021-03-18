import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.kernels import White, RBF
from gpflow.likelihoods import Gaussian

from deep_gp import DeepGP

np.random.seed(0)
tf.random.set_seed(0)


def get_data():
    Ns = 300
    Xs = np.linspace(-0.5, 1.5, Ns)[:, None]

    N, M = 50, 25
    X = np.random.uniform(0, 1, N)[:, None]
    Z = np.random.uniform(0, 1, M)[:, None]
    f_step = lambda x: 0. if x < 0.5 else 1.
    Y = np.reshape([f_step(x) for x in X], X.shape) + np.random.randn(
        *X.shape) * 1e-2
    return Xs, X, Y, Z


def make_deep_GP(num_layers, X, Y, Z):
    kernels = []
    layer_sizes = []
    for l in range(num_layers):
        kernel = RBF(lengthscales=0.2, variance=1.0) + White(variance=1e-5)
        kernels.append(kernel)
        layer_sizes.append(1)

    dgp = DeepGP(X, Y, Z, kernels, layer_sizes, Gaussian(), num_samples=100)

    # init hidden layers to be near deterministic
    for layer in dgp.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    return dgp


if __name__ == '__main__':
    Xs, X_train, Y_train, Z = get_data()
    dgp = make_deep_GP(3, X_train, Y_train, Z)
    optimizer = tf.optimizers.Adam(learning_rate=0.01, epsilon=1e-08)

    for _ in range(1500):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(dgp.trainable_variables)
            objective = -dgp.elbo((X_train, Y_train))
            gradients = tape.gradient(objective, dgp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dgp.trainable_variables))
        print(f"ELBO: {-objective.numpy()}")

    samples, _, _ = dgp.predict_all_layers(Xs, num_samples=50, full_cov=True)
    plt.plot(Xs, samples[-1].numpy()[:, :, 0].T, color='r', alpha=0.3)

    plt.title('Deep Gaussian Process')
    plt.scatter(X_train, Y_train)
    plt.show()