from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import cKDTree
import numpy as np 

class NearestPrediction(PredictivePrescription):

    def __init__(self, feats, costs, model, k):
        super().__init__(model, feats, costs)
        self.k = min(k, len(self.features)-1)

        self.tree = cKDTree(self.features)

    def _get_weights(self, x):
        distances, idx = self.tree.query(x, k=self.k, workers=-1)
        
        weights = np.zeros(len(self.features))
        weights[idx] = 1.0 / self.k
        return weights