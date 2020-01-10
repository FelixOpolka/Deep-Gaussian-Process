import tensorflow as tf
import gpflow
from gpflow.base import Module, Parameter
import numpy as np
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import triangular
import gpflow.covariances as covs

from utils import reparameterize


class Layer(Module):
    def __init__(self, input_prop_dim=None, **kwargs):
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        :param input_prop_dim: the first dimensions of X to propagate. If None (or zero) then no input prop
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.input_prop_dim = input_prop_dim

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0.0, dtype=gpflow.default_float())

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        if full_cov:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = X.shape[:3]
            X_flat = tf.reshape(X, [S*N, D])
            mean, var = self.conditional_ND(X_flat)
            num_outputs = mean.shape[-1]
            return [tf.reshape(m, [S, N, num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        # set shapes
        S = tf.shape(X)[0]
        N = tf.shape(X)[1]
        D = mean.shape[-1]
        # D = self.num_outputs

        mean = tf.reshape(mean, (S, N, D))
        if full_cov:
            var = tf.reshape(var, (S, N, N, D))
        else:
            var = tf.reshape(var, (S, N, D))

        if z is None:
            z = tf.random.normal(tf.shape(mean), dtype=gpflow.default_float())
        samples = reparameterize(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [tf.shape(X)[0], tf.shape(X)[1], self.input_prop_dim]
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = tf.concat([X_prop, samples], 2)
            mean = tf.concat([X_prop, mean], 2)

            if full_cov:
                shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[1], tf.shape(var)[3])
                zeros = tf.zeros(shape, dtype=gpflow.default_float())
                var = tf.concat([zeros, var], 3)
            else:
                var = tf.concat([tf.zeros_like(X_prop), var], 2)

        return samples, mean, var


class SVGPLayer(Layer):
    def __init__(self, kern, Z, num_outputs, mean_function,
                 white=False, input_prop_dim=None, **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :return:
        """
        super().__init__(input_prop_dim=input_prop_dim, **kwargs)
        self.num_inducing = Z.shape[0]

        # Inducing points prior mean
        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu, name="q_mu")
        # Square-root of inducing points prior covariance
        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        self.q_sqrt = Parameter(q_sqrt, transform=triangular(), name="q_sqrt")

        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs
        self.white = white

        if not self.white:  # initialize to prior
            Ku = self.kern.K(Z)
            Lu = np.linalg.cholesky(Ku + np.eye(Z.shape[0])*gpflow.default_jitter())
            self.q_sqrt = Parameter(np.tile(Lu[None, :, :], [num_outputs, 1, 1]), transform=triangular(), name="q_sqrt")

        self.Ku, self.Lu, self.Ku_tiled, self.Lu_tiled = None, None, None, None
        self.needs_build_cholesky = True

    def build_cholesky_if_needed(self):
        # # make sure we only compute this once
        # if self.needs_build_cholesky:
        self.Ku = covs.Kuu(self.feature, self.kern, jitter=gpflow.default_jitter())
        self.Lu = tf.linalg.cholesky(self.Ku)
        self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
        self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
        self.needs_build_cholesky = False

    def conditional_ND(self, X, full_cov=False):
        """
        Computes q(f|m, S; X, Z) as defined in eq. 6-8 in the paper. Remember
        that inducing point prior means are set to 0.
        :param X:
        :param full_cov:
        :return:
        """
        self.build_cholesky_if_needed()

        Kuf = covs.Kuf(self.feature, self.kern, X)

        # Compute the alpha term
        alpha = tf.linalg.triangular_solve(self.Lu, Kuf, lower=True)
        if not self.white:
            alpha = tf.linalg.triangular_solve(tf.transpose(self.Lu), alpha, lower=False)

        f_mean = tf.matmul(alpha, self.q_mu, transpose_a=True)
        f_mean = f_mean + self.mean_function(X)

        alpha_tiled = tf.tile(alpha[None, :, :], [self.num_outputs, 1, 1])

        if self.white:
            f_cov = -tf.eye(self.num_inducing, dtype=gpflow.default_float())[None, :, :]
        else:
            f_cov = -self.Ku_tiled

        if self.q_sqrt is not None:
            S = tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True) # Inducing points prior covariance
            f_cov += S

        f_cov = tf.matmul(f_cov, alpha_tiled)

        if full_cov:
            # Shape [num_latent, num_X, num_X]
            delta_cov = tf.matmul(alpha_tiled, f_cov, transpose_a=True)
            Kff = self.kern.K(X)
        else:
            # Shape [num_latent, num_X]
            delta_cov = tf.reduce_sum(alpha_tiled * f_cov, 1)
            Kff = self.kern.K_diag(X)

        # Shapes either [1, num_X] + [num_latent, num_X] or
        # [1, num_X, num_X] + [num_latent, num_X, num_X]
        f_cov = tf.expand_dims(Kff, 0) + delta_cov
        f_cov = tf.transpose(f_cov)

        return f_mean, f_cov

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, Kuu), independently
        for each GP. Uses formula for computing KL divergence between two
        multivariate normals, which in this case is:
        KL = 1/2 [log|Kuu| - log|S| - M + tr(Kuu^(-1)S) + m^T Kuu^(-1) m.
        """
        self.build_cholesky_if_needed()

        # constant dimensionality term
        KL = -0.5 * self.num_outputs * self.num_inducing
        # log of determinant of S. Uses that sqrt(det(X)) = det(X^(1/2)) and
        # that the determinant of a upper triangular matrix (which q_sqrt is),
        # is the product of the diagonal entries (i.e. sum of their logarithm).
        KL -= 0.5 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.q_sqrt) ** 2))

        if not self.white:
            # log of determinant of Kuu. Uses that determinant of triangular
            # matrix is product of diagonal entries and that 0.5*|Kuu| =
            # 0.5*|LL^T| = |L|.
            KL += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
            # computes m^T Kuu^(-1) m, which is scalar or rather has shape
            # [num_outputs]. cholesky_solve expects the Cholesky decomposition
            # of the left side (i.e. as first argument), therefore Lu is used
            # instead of Kuu.
            Kinv_m = tf.linalg.cholesky_solve(self.Lu, self.q_mu)
            KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL


