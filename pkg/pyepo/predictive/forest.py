from pyepo.predictive.pred import PredictivePrescription
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pyepo import EPO

class RandomForestPrescription(PredictivePrescription):
    def __init__(self, feats, costs, model, n_est, depth, random_state=None):
        super().__init__(model)
        self.features = feats
        self.costs = costs

        rf_model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=depth,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_model.fit(feats, costs)

        self.weigth_model = rf_model

    def _get_weights(self, x):
        T = len(self.weigth_model.estimators_)
        N = len(self.features)
        weights = np.zeros(N)

        for tree in self.weigth_model.estimators_:
            leaf_x = tree.apply([x])[0]
            leaf_train = tree.apply(self.features)
            same_leaf = (leaf_train == leaf_x)
            idx = np.where(same_leaf)[0]
            if len(idx) > 0:
                weights[idx] += 1.0 / (T * len(idx))
        return weights
