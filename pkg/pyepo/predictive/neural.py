from pyepo.predictive.pred import PredictivePrescription
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from pyepo.data.dataset import optDatasetPP
from pyepo.func.surrogate import SFGE
from pyepo.predictive.utils import EarlyStopper
import numpy as np
import time as time

from pkg.pyepo import EPO

class NeuralPrediction(PredictivePrescription):

    def __init__(self, feats, costs, weight_model, model, verbose = False):
        self.features = feats
        self.costs = costs
        self.weight_model: nn.Module = weight_model
        self.verbose = verbose
        super().__init__(model)

    def _get_weights(self, x, features=None):
        if features is None:
            features = self.features

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)           # [1, D]
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]

        x = x.to(self.weight_model.net[0].weight.device)
        features = features.to(self.weight_model.net[0].weight.device)

        weights = self.weight_model(x, features)
        return weights
    
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
    

    def train_model(self, epochs=100, batch_size=32, lr=1e-3, val_split=0.2, calc_regret : bool = False):
        X_train, X_val, y_train, y_val = train_test_split(
            self.features, self.costs, test_size=val_split, random_state=0
        )

        optimizer = optim.Adam(self.weight_model.parameters(), lr=lr)

        loss_fn = SFGE
        S = int(0.8*len(self.features)) # S for backward calculation

        train_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_train, y_train),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_val, y_val),
            batch_size=batch_size, shuffle=False
        )

        feats_full_data = torch.FloatTensor(X_train)
        sols_full_data = torch.FloatTensor(train_loader.dataset.get_sols()[0])
        objs_full_data = torch.FloatTensor(train_loader.dataset.get_sols()[1])

        if torch.cuda.is_available():
            feats_full_data = feats_full_data.cuda()
            sols_full_data = sols_full_data.cuda()
            objs_full_data = objs_full_data.cuda()

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
                loss = loss_fn(weights, c, y_obj, data_sols, self.model, S)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * -1
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
                
                    val_loss += loss_fn(weights, c, y_obj, sols, self.model, S).item() * -1

                if calc_regret:
                    regret_loss = regret_loss / opt_sum
                val_loss = val_loss / len(val_loader)
            if self.verbose:
                print(f"Epoch {epoch+1:03d}: train={train_loss:.4f}, val={val_loss:.4f}, regret_val_loss={regret_loss:.10f}")
            
            if early_stopper.step(val_loss, self.weight_model):
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}. Restored best weights.")
                break

