from pyepo.predictive.pred import PredictivePrescription
import numpy as np


class SAA(PredictivePrescription):
    def __init__(self, feats, costs, model, seed=None):
        super().__init__(model, feats, costs, seed)
    
    def _get_weights(self, x):
        return np.ones(len(self.features), dtype=float) / len(self.features)