import time
from pyepo.predictive.pred import PredictivePrescription, Predictor
from pyepo.predictive.weight_predictor import DotProductWeightPredictor
from pyepo.predictive.neural import NeuralPrediction, GroupedNeuralPrediction
from pyepo import EPO
from pyepo.model.opt import optModel
from enum import Enum
import itertools
import numpy as np
from pyepo.predictive.pool_solve import solve_in_pass

import copy

class WeightingTypeFunction(Enum):
    NEURAL = "neural"
    NEURAL_GROUPED = "neural_grouped"
    NEURAL_DFL = "neural_dfl"
    NEAREST_NEIGHBOUR = "nearest_neighbour"
    RANDOM_FOREST = "random_forest"
    LOESS = "loess"
    KERNEL = "kernel"
    RKERNEL = "rkernel"
    CART = "cart"
    SAA = "saa" 

def test_model(prediction_model: Predictor, opt_model: optModel, x_test, c_test, m_test=None, processes=1, pool=None):
    loss = 0
    optsum = 0

    _, true_objs = solve_in_pass(c_test, opt_model, processes=processes, pool=pool)
    for x, true_cost, m, true_obj in zip(x_test, c_test, m_test if m_test is not None else [None]*len(x_test), true_objs):
        pred_sol, _ = prediction_model.optimize(x, m)

        opt_model.setObj(true_cost)

        pred_obj = opt_model.cal_obj(true_cost, pred_sol)

        if opt_model.modelSense == EPO.MINIMIZE:
            loss += pred_obj - true_obj
        if opt_model.modelSense == EPO.MAXIMIZE:
            loss += true_obj - pred_obj

        optsum += abs(true_obj)

    return loss/(optsum + 1e-7)    


def finetune_predictive_prescription(
    model_cls: PredictivePrescription,
    x_train,
    c_train,
    x_val,
    c_val,
    optmodel,
    param_grid,
    model_kwargs=None,
    m_val=None,
    seed=None,
    mutli_processing=None
):
    if model_kwargs is None:
        model_kwargs = {}

    best_score = np.inf
    best_params = None

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        model = model_cls(
            x_train,
            c_train,
            optmodel,
            seed=seed,
            **params,
            **model_kwargs,
        )

        score = test_model(model, optmodel, x_val, c_val, m_val, processes=mutli_processing[0], pool=mutli_processing[1]) if mutli_processing is not None else test_model(model, optmodel, x_val, c_val, m_val)

        if score < best_score:
            best_score = score
            best_params = params

    feats = np.concatenate((x_train, x_val), axis=0)
    costs = np.concatenate((c_train, c_val), axis=0)
    start_time = time.perf_counter()
    
    best_model = model_cls(feats, costs, optmodel, seed=seed, **best_params, **model_kwargs)
    
    end_time = time.perf_counter()
    
    training_info = {"training_time": end_time - start_time,
                     "best_parameters": best_params}

    return best_model, training_info

def finetune_neural_prescription(
    feats,
    costs,
    optmodel,
    arch_param_grid,
    train_param_grid,
    loss_type,
    grouped: bool = False,
    m_train = None,
    m_val = None,
    seed = None,
):

    best_score = np.inf
    best_params = None
    best_model = None

    arch_keys = list(arch_param_grid.keys())
    arch_vals = list(arch_param_grid.values())

    train_keys = list(train_param_grid.keys())
    train_vals = list(train_param_grid.values())

    for arch_combo in itertools.product(*arch_vals):
        arch_params = dict(zip(arch_keys, arch_combo))

        for train_combo in itertools.product(*train_vals):
            train_params = dict(zip(train_keys, train_combo))

            weight_model = DotProductWeightPredictor(
                feats.shape[-1],
                **arch_params
            )

            start_time = time.perf_counter()
            if grouped:
                predictor = GroupedNeuralPrediction(
                    feats,
                    costs,
                    optmodel,
                    weight_model,
                    seed=seed
                )
            else:
                predictor = NeuralPrediction(
                    feats,
                    costs,
                    optmodel,
                    weight_model,
                    seed=seed
                )

            val_loss, train_info = predictor.train_model(
                loss_type=loss_type,
                **train_params
            )
            end_time = time.perf_counter()

            if val_loss < best_score:
                best_score = val_loss
                best_params = {**arch_params, **train_params}
                best_model = predictor
                weight_model.to("cpu")
                info = {"training_time": end_time - start_time,
                        "best_parameters": best_params,
                        **train_info}
            else:
                del predictor
                del weight_model

    print("Best params:", best_params)
    return best_model, info


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state_dict = None

    def step(self, validation_loss, model):
        if validation_loss < self.min_validation_loss * (1 - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # restore best weights and stop
                if self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                return True  # stop training
            return False   