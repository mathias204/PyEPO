from pyepo.predictive.pred import PredictivePrescription
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import numpy as np
from pyepo import EPO


class SAA(PredictivePrescription):
    def __init__(self, feats, costs, model):
        super().__init__(model)
        self.features = feats
        self.costs = costs
    
    def _get_weights(self, x):
        return np.ones(len(self.features), dtype=float) / len(self.features)