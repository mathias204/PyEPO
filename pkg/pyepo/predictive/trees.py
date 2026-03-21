from pyepo.predictive.pred import PredictivePrescription
import numpy as np
from sklearn import tree

class CartPrescription(PredictivePrescription):
    def __init__(self, feats, costs, model, random_state=None):
        super().__init__(model)
        self.features = feats
        self.costs = costs
        self.random_state = random_state

        dtr = tree.DecisionTreeRegressor(random_state=self.random_state)
        self.weight_model = dtr.fit(feats, costs)

    def _get_weights(self, x):
        N = len(self.features)
        weights = np.zeros(N)

        leaf_x = self.weight_model.apply([x])[0]
        leaf_train = self.weight_model.apply(self.features)
        same_leaf = (leaf_train == leaf_x)
        idx = np.where(same_leaf)[0]
        if len(idx) > 0:
            weights[idx] += 1.0 / len(idx)
        return weights

    