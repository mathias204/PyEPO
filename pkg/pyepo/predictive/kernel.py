from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import numpy as np
from pyepo import EPO

class KernelPrescription(PredictivePrescription):
    def __init__(self, feats, costs, model, k, kernel, random_state=None):
        super().__init__(model)
        self.random_state = random_state
        self.k = min(k, len(feats))
        self.kernel = kernel

        self.features = feats
        self.costs = costs

    @staticmethod
    def _naive_kernel(x):
        norms = np.linalg.norm(x, axis=1)
        return norms <= 1

    @staticmethod
    def _epanechnikov_kernel(x):
        norms = np.linalg.norm(x, axis=1)
        mask = norms <= 1

        results = np.zeros(len(x))
        results[mask] = 1 - norms[mask] ** 2

        return results

    @staticmethod
    def _tricubic_kernel(x):
        norms = np.linalg.norm(x, axis=1)
        mask = norms <= 1

        results = np.zeros(len(x))
        results[mask] = (1 - norms[mask] ** 3) ** 3

        return results

    def _get_weights(self, x):
        dists = distance.cdist([x], self.features, metric="euclidean").flatten()
        h_N = np.partition(dists, self.k - 1)[self.k - 1]
        h_N *= (1 + 1e-8) # Otherwise if k = 1, no points are within the bandwith


        if h_N == 0:
            zero_mask = (dists == 0)
            count_zero = np.count_nonzero(zero_mask)
            if count_zero > 0:
                return zero_mask.astype(float) / float(count_zero)
            # Defensive fallback: no positive bandwidth and no exact matches.
            return np.ones(len(self.features), dtype=float) / float(len(self.features))


        delta_x = self.features - x

        kernel_outputs = self.kernel(delta_x / h_N)
        kernel_sum = np.sum(kernel_outputs)

        if kernel_sum > 0:
            return kernel_outputs.astype(float) / kernel_sum
        
        # If the kernel assigns zero mass everywhere, fall back to global uniform weights.
        return np.ones(len(self.features), dtype=float) / float(len(self.features))


class RecursiveKernelPrescription(KernelPrescription):
    def __init__(self, feats, costs, model, k, kernel, random_state=None):
        super().__init__(feats, costs, model, k, kernel, random_state)

        pairwise_dists = distance.cdist(self.features, self.features, metric="euclidean")
        np.fill_diagonal(pairwise_dists, np.inf)
        pairwise_dists = pairwise_dists
        self._h_i = np.partition(pairwise_dists, self.k - 1, axis=1)[:, self.k - 1] * (1 + 1e-8)

    def _get_weights(self, x):
        delta_x = self.features - x
        scaled = delta_x / self._h_i[:, None]

        kernel_outputs = self.kernel(scaled)
        kernel_sum = np.sum(kernel_outputs)

        if kernel_sum > 0:
            return kernel_outputs.astype(float) / kernel_sum

        return np.ones(len(self.features), dtype=float) / len(self.features)

