from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
import numpy as np 

class NearestPrediction(PredictivePrescription):

    def __init__(self, feats, costs, model, k):
        super().__init__(model, feats, costs)
        self.k = min(k, len(self.features)-1)

    def _get_weights(self, x):
        dists = distance.cdist([x], self.features, metric="euclidean").flatten()
        idx = np.argpartition(dists, self.k)[:self.k]
        weights = np.zeros(len(self.features))
        weights[idx] = 1.0 / self.k
        return weights