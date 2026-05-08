import time

from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
import numpy as np
from scipy.spatial import cKDTree
from scipy import linalg

class LOESS(PredictivePrescription):

    def __init__(self, feats, costs, model, k):
        super().__init__(model, feats, costs)

        self.k = min(k, len(self.features))
        self.tree = cKDTree(self.features)

    def _get_weights(self, x):
        dists, idx = self.tree.query(x, k=self.k)
        
        dists = np.atleast_1d(dists)
        idx = np.atleast_1d(idx)

        h_N = dists[-1]

        if h_N == 0:
            weights = np.zeros(len(self.features))
            zero_mask = dists == 0
            weights[idx[zero_mask]] = 1.0 / np.sum(zero_mask)
            return weights

        local_features = self.features[idx]

        u = dists / h_N
        local_k_val = (1.0 - u**3)**3

        local_delta_x = local_features - x

        Xi = (local_delta_x.T * local_k_val) @ local_delta_x
        v = local_k_val @ local_delta_x

        # Regularize the diagonal in-place for numerical stability
        # This makes the symmetric matrix strictly positive-definite
        Xi.flat[::Xi.shape[0] + 1] += 1e-8

        # Solve Xi * w = v directly using Cholesky decomposition
        # This replaces the expensive SVD from np.linalg.pinv
        v_Xi_inv = linalg.solve(Xi, v, assume_a='pos')

        T = local_delta_x @ v_Xi_inv

        local_weights = local_k_val * np.maximum(1.0 - T, 0.0)
        weight_sum = np.sum(local_weights)

        # Generating a 50k dense array per query point exhausts memory bandwidth
        weights = np.zeros(len(self.features))
        
        if weight_sum > 0:
            weights[idx] = local_weights / weight_sum
        else:
            weights[idx] = 1.0 / self.k

        return weights
    
    # def _get_weights2(self, x):
    #     dists = distance.cdist([x], self.features, metric="euclidean").flatten()
    #     h_N = np.partition(dists, self.k - 1)[self.k - 1]

    #     if h_N == 0:
    #         weights = np.zeros(len(self.features))
    #         weights[dists == 0] = 1.0 / np.sum(dists == 0)
    #         return weights

    #     # Tri-cube kernel computation
    #     u = dists / h_N
    #     mask = dists <= h_N
    #     k_val = np.zeros(len(self.features))
    #     k_val[mask] = (1 - u[mask]**3)**3

    #     # Delta matrix (X - x), shape (n, d)
    #     delta_x = self.features - x

    #     # Matrix Xi(x): sum_i k_i(x)(x^i - x)(x^i - x)^T
    #     # delta_x.T * k_val scales each row, followed by matrix multiplication
    #     Xi = (delta_x.T * k_val) @ delta_x

    #     # Vector v(x): sum_j k_j(x)(x^j - x)^T
    #     v = k_val @ delta_x

    #     # Use pseudo-inverse for numerical stability in case Xi is singular
    #     Xi_inv = np.linalg.pinv(Xi)

    #     # Compute the inner product term for all i simultaneously
    #     v_Xi_inv = v @ Xi_inv
    #     T = delta_x @ v_Xi_inv

    #     weights = k_val * np.maximum(1 - T, 0)

    #     weight_sum = np.sum(weights)
    #     if weight_sum > 0:
    #         weights /= weight_sum
    #     else:
    #         idx = np.argpartition(dists, self.k)[:self.k-1]
    #         weights = np.zeros(len(self.features))
    #         weights[idx] = 1.0 / self.k

    #     return weights