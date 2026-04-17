#!/usr/bin/env python
# coding: utf-8
"""
Abstract predictive prescription model
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from pyepo.model.opt import optModel

class PredictivePrescription(ABC):
    """
    This is an abstract class for predicitive prescription model
    """
    def __init__(self, model, features, costs):
        self.features_unadjusted = features
        self.costs_unadjusted = costs
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
            costs = costs.reshape(-1) 
        self.features = features
        self.costs = costs
        self.model: optModel = model
    
    @abstractmethod
    def _get_weights(self, x):
        """
        An abstract method to gather the weights for the prediction
        """
        raise NotImplementedError
    
    def _create_numpy_weights(self, weights):
        if isinstance(weights, torch.Tensor):
            if weights.is_cuda:
                weights = weights.detach().cpu()
            else:
                weights = weights.detach()

            if weights.dim() == 2 and weights.size(0) == 1:
                weights = weights.squeeze(0)

            weights = weights.numpy()
        return weights

    
    def _optimize_shared(self, x, m = None):
        if m is not None:
            constraints = self.model.setM(m)

        W = np.ndarray((len(x), len(self.features)))
        with torch.no_grad():
            for i,x_i in enumerate(x):
                weights = self._get_weights(x_i)
                weights = self._create_numpy_weights(weights)
                if not np.isclose(np.sum(weights), 1.0):
                    raise RuntimeError(f"Weights do not sum to 1.0, got sum {np.sum(weights)}")
                W[i] = weights       
                    
        # Optimize
        self.model.setWeightObj(W, self.costs)
        sol, obj = self.model.solve()

        if m is not None:
            self.model.removeM(constraints)

        if isinstance(sol, torch.Tensor):
            sol = sol.detach().cpu().numpy()

        return sol, obj


    def optimize(self, x, m=None): 
        # Predict
        if x.ndim == 2:
            return self._optimize_shared(x, m)

        with torch.no_grad():
            weights = self._get_weights(x)

        if isinstance(weights, torch.Tensor):
            if weights.is_cuda:
                weights = weights.detach().cpu()
            else:
                weights = weights.detach()

            if weights.dim() == 2 and weights.size(0) == 1:
                weights = weights.squeeze(0)

            weights = weights.numpy()

        if not np.isclose(np.sum(weights), 1.0):
            raise RuntimeError(f"Weights do not sum to 1.0, got sum {np.sum(weights)}")
                        

        # Optimize
        self.model.setWeightObj(weights, self.costs)
        sol, obj = self.model.solve()

        if isinstance(sol, torch.Tensor):
            sol = sol.detach().cpu().numpy()

        return sol, obj