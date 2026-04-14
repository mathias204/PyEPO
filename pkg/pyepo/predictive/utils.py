from pyepo.predictive.pred import PredictivePrescription
from pyepo.predictive.neural import NeuralPrediction
from pyepo import EPO
from pyepo.model.opt import optModel
from enum import Enum
import itertools
from sklearn.model_selection import train_test_split
import numpy as np

class WeightingTypeFunction(Enum):
    NEURAL = "neural"
    NEAREST_NEIGHBOUR = "nearest_neighbour"
    RANDOM_FOREST = "random_forest"
    LOESS = "loess"
    KERNEL = "kernel"
    RKERNEL = "rkernel"
    CART = "cart"
    SAA = "saa" 

def test_model(prediction_model: PredictivePrescription, opt_model: optModel, x_test, c_test):
    # TODO: can be made a little more efficient by batching, only setting objective and solving can't be batched
    loss = 0
    optsum = 0

    for x, true_cost in zip(x_test, c_test):

        pred_sol, _ = prediction_model.optimize(x)

        opt_model.setObj(true_cost)
        _, true_obj = opt_model.solve()

        pred_obj = opt_model.cal_obj(true_cost, pred_sol)

        if opt_model.modelSense == EPO.MINIMIZE:
            loss += pred_obj - true_obj
        if opt_model.modelSense == EPO.MAXIMIZE:
            loss += true_obj - pred_obj

        optsum += abs(true_obj)

    return loss/(optsum + 1e-7)    


def finetune_predictive_prescription(
    model_cls: PredictivePrescription,
    feats,
    costs,
    optmodel,
    param_grid,
    test_size=0.2,
    random_state=None,
    model_kwargs=None,
):
    x_train, x_val, c_train, c_val = train_test_split(
        feats, costs, test_size=test_size, random_state=random_state
    )

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
            **params,
            **model_kwargs,
        )

        score = test_model(model, optmodel, x_val, c_val)

        if score < best_score:
            best_score = score
            best_params = params

    return model_cls(feats, costs, optmodel, **best_params, **model_kwargs)

def finetune_neural_prescription(
    feats,
    costs,
    optmodel,
    weight_model_class,
    arch_param_grid,
    train_param_grid,
    loss_type
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

            weight_model = weight_model_class(
                feats.shape[1],
                **arch_params
            )

            predictor = NeuralPrediction(
                feats,
                costs,
                optmodel,
                weight_model,
            )

            val_loss = predictor.train_model(
                loss_type=loss_type,
                **train_params
            )

            if val_loss < best_score:
                best_score = val_loss
                best_params = {**arch_params, **train_params}
                best_model = predictor

    print("Best params:", best_params)
    return best_model