from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import numpy as np
from pyepo import EPO

class KernelPrescription(PredictivePrescription):
    def __init__(self, feats, costs, k, model, random_state=None):
        super().__init__(model)
        self.random_state = random_state
        self.k = k
        self.kernel = self._optimize_model(feats, costs)

        # Must be done after _optimize_model
        self.features = feats
        self.costs = costs

    def _naive_kernel(self, x):
        norms = np.linalg.norm(x, axis=1)
        return norms <= 1

    def _epanechnikov_kernel(self, x):
        norms = np.linalg.norm(x, axis=1)
        mask = norms <= 1

        results = np.zeros(len(self.features))
        results[mask] = 1 - norms[mask] ** 2

        return results

    def _tricubic_kernel(self, x):
        norms = np.linalg.norm(x, axis=1)
        mask = norms <= 1

        results = np.zeros(len(self.features))
        results[mask] = (1 - norms[mask] ** 3) ** 3

        return results

    def _optimize_model(self, feats, costs):
        X_train, X_val, y_train, y_val = train_test_split(
            feats, costs, test_size=0.2, random_state=self.random_state
        )

        self.features = X_train
        self.costs = y_train

        kernels = [
            self._naive_kernel, self._epanechnikov_kernel, self._tricubic_kernel
        ]

        best_score = np.inf
        best_kernel = None

        for kernel in kernels:
            self.kernel = kernel

            loss = 0
            optsum = 0
            for x, c in zip(X_val, y_val):
                sol, obj = self.optimize(x)

                self.model.setObj(c)
                _, true_obj = self.model.solve()
                pred_obj = self.model.cal_obj(c, sol)

                if self.model.modelSense == EPO.MINIMIZE:
                    loss += pred_obj - true_obj
                if self.model.modelSense == EPO.MAXIMIZE:
                    loss += true_obj - pred_obj

                optsum += abs(true_obj)

            score = loss / (optsum + 1e-7)
            if score < best_score:
                best_score = score
                best_kernel = kernel

        return best_kernel

    def _get_weights(self, x):
        dists = distance.cdist([x], self.features, metric="euclidean").flatten()
        h_N = np.partition(dists, self.k - 1)[self.k - 1]

        delta_x = self.features - x

        kernel_outputs = self.kernel(delta_x / h_N)
        kernel_sum = np.sum(kernel_outputs)

        return kernel_outputs.astype(float) / kernel_sum


class RecursiveKernelPrescription(KernelPrescription):
    def __init__(self, feats, costs, k, model, random_state=None):
        super().__init__(feats, costs, k, model, random_state)

    def _get_weights(self, x):
        dists = distance.cdist(self.features, self.features, metric="euclidean")
        np.fill_diagonal(dists, np.inf)

        h_i = np.partition(dists, self.k - 1, axis=1)[:, self.k - 1]

        delta_x = self.features - x
        scaled = delta_x / h_i[:, None]

        kernel_outputs = self.kernel(scaled)
        kernel_sum = np.sum(kernel_outputs)

        if kernel_sum > 0:
            return kernel_outputs.astype(float) / kernel_sum

        return np.ones(len(self.features), dtype=float) / len(self.features)
