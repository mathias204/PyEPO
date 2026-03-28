from pyepo.predictive.pred import PredictivePrescription
import numpy as np


class SAA(PredictivePrescription):
    def __init__(self, feats, costs, model):
        super().__init__(model, feats, costs)
    
    def _get_weights(self, x):
        return np.ones(len(self.features), dtype=float) / len(self.features)