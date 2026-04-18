from pyepo.predictive.neural import LossType
from pyepo.func.surrogate import SPOPlus
from pyepo.predictive.dfl import DFLPredictor
from pyepo.predictive.pred import PredictivePrescription, Predictor
from pyepo.predictive.neural import NeuralPrediction, GroupedNeuralPrediction
from pyepo import EPO
from pyepo.model.opt import optModel
from enum import Enum
import itertools
import numpy as np
import torch

from pyepo.data.dataset import optDataset
from torch import optim
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

def test_model(prediction_model: Predictor, opt_model: optModel, x_test, c_test, m_test=None):
    # TODO: can be made a little more efficient by batching, only setting objective and solving can't be batched
    loss = 0
    optsum = 0

    for x, true_cost, m in zip(x_test, c_test, m_test if m_test is not None else [None]*len(x_test)):
        pred_sol, _ = prediction_model.optimize(x, m)

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
    x_train,
    c_train,
    x_val,
    c_val,
    optmodel,
    param_grid,
    model_kwargs=None,
    m_val=None
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
            **params,
            **model_kwargs,
        )

        score = test_model(model, optmodel, x_val, c_val, m_val)

        if score < best_score:
            best_score = score
            best_params = params

    feats = np.concatenate((x_train, x_val), axis=0)
    costs = np.concatenate((c_train, c_val), axis=0)

    return model_cls(feats, costs, optmodel, **best_params, **model_kwargs)

def finetune_neural_prescription(
    feats,
    costs,
    optmodel,
    weight_model_class,
    arch_param_grid,
    train_param_grid,
    loss_type,
    grouped: bool = False,
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
                feats.shape[-1],
                **arch_params
            )

            if grouped:
                predictor = GroupedNeuralPrediction(
                    feats,
                    costs,
                    optmodel,
                    weight_model,
                )
            else:
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


def finetune_neural_dfl(
    x_train,
    y_train,
    x_val,
    y_val,
    optmodel,
    neural_model_class,
    arch_param_grid,
    train_param_grid,
    loss_type,
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

            neural_model = neural_model_class(
                x_train.shape[-1],
                y_train.shape[-1],
                **arch_params
            )

            val_loss = train_neural_dfl(x_train, y_train, x_val, y_val, optmodel, neural_model, arch_params,loss_type=loss_type ,**train_params)


            if val_loss < best_score:
                predictor = DFLPredictor(optmodel, neural_model)
                best_score = val_loss
                best_params = {**arch_params, **train_params}
                best_model = predictor

    print("Best params:", best_params)
    return best_model

def train_neural_dfl(x_train, y_train, x_val, y_val, model, neural_model, arch_params, batch_size, epochs, loss_type, lr, verbose=False):
    train_loader = torch.utils.data.DataLoader(
        optDataset(model, x_train, y_train),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        optDataset(model, x_val, y_val),
        batch_size=batch_size, shuffle=False
    )

    early_stopper = EarlyStopper(5, 0)
    optimizer = optim.Adam(neural_model.parameters(), lr=lr)


    if loss_type == LossType.SPO:
        spo_plus = SPOPlus(model)

    if torch.cuda.is_available():
        neural_model = neural_model.cuda()

    for epoch in range(epochs):
        neural_model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            x, y, sol, obj = data

            if torch.cuda.is_available():
                x, y, sol, obj = x.cuda(), y.cuda(), sol.cuda(), obj.cuda()
            
            # forward pass
            y_hat = neural_model(x)             # [B, N]
            
            if loss_type == LossType.SPO:
                loss = spo_plus(y_hat, y, sol, obj)
            else:
                raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.DER.")
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        # validation
        neural_model.eval()
        with torch.no_grad():
            val_loss = 0.0  
            for i, data in enumerate(val_loader):
                x, y, sol, obj = data

                if torch.cuda.is_available():
                    x, y, sol, obj = x.cuda(), y.cuda(), sol.cuda(), obj.cuda()

                y_hat = neural_model(x)

                if loss_type == LossType.SPO:
                    val_loss = spo_plus(y_hat, y, sol, obj)
                else:
                    raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.DER.")
            
            val_loss = val_loss / len(val_loader)

        if verbose:
            print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if early_stopper.step(val_loss, neural_model):
            print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}")
            if verbose:
                print(f"Early stopping at epoch {epoch+1}. Restored best weights.")
            break

        if epoch == epochs - 1:
            print(f"Finished training for {epochs} epochs. Restoring best weights.")
            print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}")
        
    neural_model.eval()
    return val_loss



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state_dict = None

    def step(self, validation_loss, model):
        if validation_loss < self.min_validation_loss - self.min_delta:
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