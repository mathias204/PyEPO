from pyepo.predictive.pred import PredictivePrescription
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestPrescription(PredictivePrescription):
    def __init__(self, feats, costs, model, n_est, depth, random_state=None):
        super().__init__(model, feats, costs)
        rf_model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(self.features, self.costs)

        self.weigth_model = rf_model

        self._precompute_leaf_weights()

    def _precompute_leaf_weights(self):
        self._train_leaf_indices = np.array([
            tree.apply(self.features) for tree in self.weigth_model.estimators_
        ])

    def _get_weights(self, x):
        T = len(self.weigth_model.estimators_)
        N = len(self.features)
        weights = np.zeros(N)

        # Get leaf index for the input 'x' across all trees: Shape (T,)
        x_leaf_indices = np.array([
            tree.apply([x])[0] for tree in self.weigth_model.estimators_
        ])

        for t in range(T):
            # Use the precomputed indices for tree 't'
            leaf_train = self._train_leaf_indices[t]
            same_leaf = (leaf_train == x_leaf_indices[t])
            
            idx = np.where(same_leaf)[0]
            count = len(idx)
            if count > 0:
                weights[idx] += 1.0 / (T * count)
                
        return weights