from pyepo.predictive.pred import PredictivePrescription
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from pyepo.data.dataset import optDatasetPP
import numpy as np
import time as time
from pyepo import EPO
from enum import Enum
import copy
from pyepo.func.surrogate import SFGE, novel, SPOPlus

class LossType(Enum):
    SFGE = 1
    SPO = 2
    NOVEL = 3

class NeuralPrediction(PredictivePrescription):

    def __init__(self, feats, costs, model, weight_model, verbose = False):
        super().__init__(model, feats, costs)
        self.weight_model: nn.Module = weight_model
        self.verbose = verbose

    def _get_weights_shared(self, x, features):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)           # [1, X, D]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]


        device = next(self.weight_model.parameters()).device
        x = x.to(device)
        features = features.to(device)

        weights = self.weight_model(x, features)
        return weights


    def _get_weights(self, x, features=None):
        if features is None:
            features = self.features

        if self.features_unadjusted.ndim == 3:
            return self._get_weights_shared(x, features)

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)           # [1, D]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]

        device = next(self.weight_model.parameters()).device
        x = x.to(device)
        features = features.to(device)

        weights = self.weight_model(x, features)
        return weights
    
    def _optimize_shared(self, x):
        with torch.no_grad():
            W = self._get_weights(x)
            
            sums = W.sum(dim=-1)
    
            if not torch.allclose(sums, torch.ones_like(sums)):
                raise RuntimeError("Weights do not sum to 1.0 along the N dimension.")
            
        W = W.detach().cpu()
        W = W.squeeze(0)
        # Optimize
        self.model.setWeightObj(W, self.costs)
        sol, obj = self.model.solve()

        if isinstance(sol, torch.Tensor):
            sol = sol.detach().cpu().numpy()

        return sol, obj
    
    def _calculate_regret(self, weights, costs, true_objs, data_sols):
        preds = []
        for i, weight in enumerate(weights):
            self.model.setWeightObj(weight, costs)
            sol, obj = self.model.solve()
            if isinstance(sol, torch.Tensor):
                sol = sol.detach().cpu().numpy()
            else:
                sol = np.array(sol)
            preds.append(sol)

        preds = np.array(preds)
        true_objs_np = true_objs.squeeze().cpu().numpy()  

        realised_obj = self.model.cal_obj(costs, preds)

        if self.model.modelSense == EPO.MINIMIZE:
            regret = realised_obj - true_objs_np
        else:
            regret = true_objs_np - realised_obj

        return regret
    
    def _novel_loss(self, weights, costs, true_objs, data_sols):
        loss = novel(weights, costs, true_objs, data_sols, self.model)
        return loss
    
    def _sfge_loss(self, weights, costs, true_objs, data_sols, S) -> torch.Tensor:
        loss = SFGE(weights, costs, true_objs, data_sols, self.model, S)
        return loss
    
    def _spo_loss(self, spo_plus, weights, costs_batch, true_costs, true_sols, true_objs) -> torch.Tensor:
        if self.features_unadjusted.ndim == 3:
            y_hat = torch.einsum('bxn,bn->bx', weights, costs_batch)
        else:
            y_hat = (weights.unsqueeze(-1) * costs_batch).sum(dim=1)

        return spo_plus(y_hat, true_costs, true_sols, true_objs)    
    

    def train_model(self, epochs=100, batch_size=32, lr=1e-3, val_split=0.11, calc_regret : bool = False, loss_type : LossType = LossType.SFGE):
        X_train, X_val, y_train, y_val = train_test_split(
            self.features_unadjusted, self.costs_unadjusted, test_size=val_split, random_state=0
        )

        optimizer = optim.Adam(self.weight_model.parameters(), lr=lr)

        S = int(0.8*len(self.features)) # S for backward calculation

        if loss_type == LossType.SPO:
            spo_plus = SPOPlus(self.model)

        train_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_train, y_train),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_val, y_val),
            batch_size=batch_size, shuffle=False
        )

        feats_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[0])
        costs_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[1])
        sols_full_data = torch.FloatTensor(train_loader.dataset.get_sols()[0])
        objs_full_data = torch.FloatTensor(train_loader.dataset.get_sols()[1])

        if torch.cuda.is_available():
            feats_full_data = feats_full_data.cuda()
            sols_full_data = sols_full_data.cuda()
            objs_full_data = objs_full_data.cuda()
            costs_full_data = costs_full_data.cuda()

            self.weight_model = self.weight_model.cuda()

        early_stopper = EarlyStopper(5, 0)

        for epoch in range(epochs):
            self.weight_model.train()
            train_loss = 0.0
            opt_sum = 0.0
            for i, data in enumerate(train_loader):
                x, c, y_sol, y_obj, data_feats, data_costs, data_sols, data_objs = data

                if torch.cuda.is_available():
                    x, c, y_sol, y_obj, data_feats, data_costs, data_sols, data_objs = x.cuda(), c.cuda(), y_sol.cuda(), y_obj.cuda(), data_feats.cuda(), data_costs.cuda(), data_sols.cuda(), data_objs.cuda()
                # forward pass
                weights = self._get_weights(x, data_feats)             # [B, N]
                if loss_type == LossType.SFGE:
                    loss = self._sfge_loss(weights, c, y_obj, data_sols, S)
                elif loss_type == LossType.SPO:
                    loss = self._spo_loss(spo_plus, weights, data_costs, c, y_sol, y_obj)
                elif loss_type == LossType.NOVEL:
                    loss = self._novel_loss(weights, c, y_obj, data_sols)
                else:
                    raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.NOVEL.")
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()


                opt_sum += np.sum(abs(y_obj.squeeze().cpu().numpy()))

            train_loss = train_loss / len(train_loader)

            # validation
            self.weight_model.eval()
            with torch.no_grad():
                val_loss = 0.0  
                regret_loss = 0.0
                opt_sum = 0.0
                for i, data in enumerate(val_loader):
                    x, c, y_sol, y_obj, data_feats, data_costs, data_sols, data_objs = data
                    
                    if torch.cuda.is_available():
                        x, c, y_sol, y_obj, data_feats, data_costs, data_sols, data_objs = x.cuda(), c.cuda(), y_sol.cuda(), y_obj.cuda(), data_feats.cuda(), data_costs.cuda(), data_sols.cuda(), data_objs.cuda()

                    feats_batch = feats_full_data.unsqueeze(0).expand(len(x), -1, -1).contiguous()  # [B, N, D]
                    sols = sols_full_data.unsqueeze(0).expand(len(x), -1, -1).contiguous()
                    
                    weights = self._get_weights(x, feats_batch)

                    if calc_regret:
                        regret = self._calculate_regret(weights, c, y_obj, sols)
                        opt_sum += np.sum(abs(y_obj.squeeze().cpu().numpy()))
                        regret_loss += np.sum(regret).item()

                    if loss_type == LossType.SFGE:
                        val_loss += self._sfge_loss(weights, c, y_obj, sols, S).item() * -1 #TODO: is this -1 correct
                    elif loss_type == LossType.SPO:
                        costs_full_batch = costs_full_data.unsqueeze(0).expand(x.shape[0], -1)
                        val_loss += self._spo_loss(spo_plus, weights, costs_full_batch, c, y_sol, y_obj).item()
                    elif loss_type == LossType.NOVEL:
                        val_loss += self._novel_loss(weights, c, y_obj, sols).item()
                    else:
                        raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.NOVEL.")

                if calc_regret:
                    regret_loss = regret_loss / opt_sum
                val_loss = val_loss / len(val_loader)
            if self.verbose:
                print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}, regret_val_loss={regret_loss:.10f}")
            
            if early_stopper.step(val_loss, self.weight_model):
                print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}, regret_val_loss={regret_loss:.10f}")
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}. Restored best weights.")
                break

            if epoch == epochs - 1:
                print(f"Finished training for {epochs} epochs. Restoring best weights.")
                print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}, regret_val_loss={regret_loss:.10f}")
        
        self.weight_model.eval()

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