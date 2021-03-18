import gpflow
import numpy as np
from gpflow.mean_functions import Identity, Linear, Zero

from layers import SVGPLayer


def init_layers_linear(X, Y, Z, kernels, layer_sizes, mean_function=Zero(),
                       num_outputs=None, Layer=SVGPLayer, whiten=False):
    num_outputs = num_outputs or Y.shape[1]
    layers = []

    X_running, Z_running = X.copy(), Z.copy()
    for in_idx, kern_in in enumerate(kernels[:-1]):
        dim_in = layer_sizes[in_idx]
        dim_out = layer_sizes[in_idx+1]

        # Initialize mean function to be either Identity or PCA projection
        if dim_in == dim_out:
            mf = Identity()
        else:
            if dim_in > dim_out:  # stepping down, use the pca projection
                # use eigenvectors corresponding to dim_out largest eigenvalues
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T
            else:                 # stepping up, use identity + padding
                W = np.concatenate([np.eye(dim_in),
                                    np.zeros((dim_in, dim_out - dim_in))], 1)
            mf = Linear(W)
            gpflow.set_trainable(mf.A, False)
            gpflow.set_trainable(mf.b, False)

        layers.append(Layer(kern_in, Z_running, dim_out, mf, white=whiten))

        if dim_in != dim_out:
            Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)

    # final layer
    layers.append(Layer(kernels[-1], Z_running, num_outputs, mean_function,
                        white=whiten))
    return layers
