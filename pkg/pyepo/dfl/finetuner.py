import itertools
import numpy as np
from pyepo.dfl.MSE import MSEDecisionMaker
from pyepo.dfl.predictor import MLPPredictor
from pyepo.dfl.noisifier import Noisifier
from pyepo.dfl.SFGE import SFGEDecisionMaker
from pyepo.dfl.SPO import SPODecisionMaker
from pyepo.predictive.neural import LossType
import time

import torch

def dfl_finetune(
    x_train,
    y_train,
    x_val,
    y_val,
    optmodel,
    arch_param_grid,
    train_param_grid,
    loss_type: LossType,
    seed = None
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

            start_time = time.perf_counter()
            predictor = MLPPredictor(
                x_train.shape[-1],
                y_train.shape[-1],
                **arch_params
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if loss_type == LossType.SFGE:
                noisifier = Noisifier(predictor)
                dfl_maker = SFGEDecisionMaker(noisifier, optmodel, seed=seed, device=device, **train_params)
            elif loss_type == LossType.SPO:
                dfl_maker = SPODecisionMaker(predictor, optmodel, seed=seed, device=device, **train_params)
            elif loss_type == LossType.MSE:
                dfl_maker = MSEDecisionMaker(predictor, optmodel, seed=seed, device=device, **train_params)

            val_loss, train_info = dfl_maker.train_model(x_train, y_train, x_val, y_val)
            end_time = time.perf_counter()

            if val_loss < best_score:
                best_score = val_loss
                best_params = {**arch_params, **train_params}
                best_model = dfl_maker
                info = {**train_info, 
                        "training_time": end_time - start_time,
                        "best_parameters": best_params}

    print("Best params:", best_params)
    return best_model, info