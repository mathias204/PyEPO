from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
import numpy as np

class LOESS(PredictivePrescription):

    def __init__(self, feats, costs, model, k):
        super().__init__(model, feats, costs)
        
        self.k = min(k, len(self.features))


    def _get_weights(self, x):
        dists = distance.cdist([x], self.features, metric="euclidean").flatten()
        h_N = np.partition(dists, self.k - 1)[self.k - 1]

        if h_N == 0:
            weights = np.zeros(len(self.features))
            weights[dists == 0] = 1.0 / np.sum(dists == 0)
            return weights

        # Tri-cube kernel computation
        u = dists / h_N
        mask = dists <= h_N
        k_val = np.zeros(len(self.features))
        k_val[mask] = (1 - u[mask]**3)**3

        # Delta matrix (X - x), shape (n, d)
        delta_x = self.features - x

        # Matrix Xi(x): sum_i k_i(x)(x^i - x)(x^i - x)^T
        # delta_x.T * k_val scales each row, followed by matrix multiplication
        Xi = (delta_x.T * k_val) @ delta_x

        # Vector v(x): sum_j k_j(x)(x^j - x)^T
        v = k_val @ delta_x

        # Use pseudo-inverse for numerical stability in case Xi is singular
        Xi_inv = np.linalg.pinv(Xi)

        # Compute the inner product term for all i simultaneously
        v_Xi_inv = v @ Xi_inv
        T = delta_x @ v_Xi_inv

        weights = k_val * np.maximum(1 - T, 0)

        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        else:
            idx = np.argpartition(dists, self.k)[:self.k]
            weights = np.zeros(len(self.features))
            weights[idx] = 1.0 / self.k

        return weights