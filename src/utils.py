import gpflow
import tensorflow as tf
from gpflow.likelihoods import Likelihood, Gaussian


class BroadcastingLikelihood(Likelihood):
    """
    A wrapper for the likelihood to broadcast over the samples dimension.
    The Gaussian doesn't need this, but for the others we can apply reshaping
    and tiling. With this wrapper all likelihood functions behave correctly
    with inputs of shape S,N,D, but with Y still of shape N,D
    """

    def __init__(self, likelihood):
        super().__init__(likelihood.latent_dim, likelihood.observation_dim)
        self.likelihood = likelihood

        if isinstance(likelihood, Gaussian):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if not self.needs_broadcasting:
            return f(vars_SND, [tf.expand_dims(v, 0) for v in vars_ND])
        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N])

    def _variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                               vars_SND[1],
                                                                               vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])

    def _log_prob(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.log_prob(vars_SND[0],
                                                               vars_ND[0])
        return self._broadcast(f, [F], [Y])

    def _conditional_mean(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(
            vars_SND[0])
        return self._broadcast(f, [F], [])

    def _conditional_variance(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(
            vars_SND[0])
        return self._broadcast(f, [F], [])

    def _predict_mean_and_var(self, Fmu, Fvar):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(
            vars_SND[0],
            vars_SND[1])
        return self._broadcast(f, [Fmu, Fvar], [])

    def _predict_log_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_log_density(
            vars_SND[0],
            vars_SND[1],
            vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + gpflow.default_jitter()) ** 0.5
    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = gpflow.default_jitter() * tf.eye(N, dtype=gpflow.default_float())[None, None, :, :] # 11NN
        chol = tf.linalg.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


def summarize_tensor(x, title=""):
    print("-"*10, title, "-"*10, sep="")
    shape = x.shape
    print(f"Shape: {shape}")

    nans = tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int16))
    print(f"NaNs: {nans}")

    nnz = tf.reduce_sum(tf.cast(x < 1e-8, tf.int16))
    print(f"NNZ: {nnz}")

    mean = tf.reduce_mean(x)
    print(f"Mean: {mean}")
    std = tf.math.reduce_std(x)
    print(f"Std: {std}")

    min = tf.math.reduce_min(x)
    print(f"Min: {min}")
    max = tf.math.reduce_max(x)
    print(f"Max: {max}")
    print("-"*(20+len(title)))