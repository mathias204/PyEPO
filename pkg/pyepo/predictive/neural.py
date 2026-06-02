from pyepo.predictive.pred import PredictivePrescription
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from pyepo.data.dataset import optDatasetPP, optDatasetSharedPP
import numpy as np
import time as time
from pyepo import EPO
from enum import Enum
import copy
from pyepo.func.surrogate import SFGE, DER, SPOPlus
from torch.utils.data._utils.collate import default_collate
from pyepo.predictive.pool_solve import solve_in_pass
from pathos.multiprocessing import ProcessingPool
import multiprocessing as mp

class LossType(Enum):
    SFGE = 1
    SPO = 2
    DER = 3
    MSE = 4 

class NeuralPrediction(PredictivePrescription):

    def __init__(self, feats, costs, model, weight_model, verbose = False, seed=None):
        super().__init__(model, feats, costs, seed)
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
    
    def _optimize_shared(self, x, m=None):
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
    
    def _der_loss(self, weights, costs, true_objs, data_sols):
        loss = DER(weights, costs, true_objs, data_sols, self.model)
        return loss
    
    def _sfge_loss(self, weights, costs, true_objs, data_sols, S) -> torch.Tensor:
        loss = SFGE(weights, costs, true_objs, data_sols, self.model, S)
        return loss
    
    def _spo_loss(self, spo_plus, weights, costs_batch, true_costs, true_sols, true_objs) -> torch.Tensor:
        y_hat = torch.einsum('bn,bnc->bc', weights, costs_batch)
        y_hat = self.model.transform_prediction(y_hat)
        true_costs = self.model.transform_prediction(true_costs)

        if not isinstance(true_sols, torch.Tensor):
            true_sols = torch.tensor(true_sols, dtype=torch.float32, device=y_hat.device)

        return spo_plus(y_hat, true_costs, true_sols, true_objs)    
    

    def train_model(self, epochs=100, batch_size=32, lr=1e-3, val_split=0.11, calc_regret : bool = False, loss_type : LossType = LossType.SFGE):
        g = torch.Generator()
        if self.seed is not None:
            g = g.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        X_train, X_val, y_train, y_val = train_test_split(
            self.features_unadjusted, self.costs_unadjusted, test_size=val_split, random_state=self.seed
        )

        optimizer = optim.Adam(self.weight_model.parameters(), lr=lr)

        S = int(0.8*len(self.features)) # S for backward calculation

        if loss_type == LossType.SPO:
            spo_plus = SPOPlus(self.model, processes=0)

        train_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_train, y_train),
            batch_size=batch_size, shuffle=True, generator=g,
            collate_fn=custom_collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            optDatasetPP(self.model, X_val, y_val),
            batch_size=batch_size, shuffle=False,
            collate_fn=custom_collate_fn
        )

        feats_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[0])
        costs_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[1])
        sols_full_data = train_loader.dataset.get_sols()[0]
        objs_full_data = torch.FloatTensor(train_loader.dataset.get_sols()[1])

        if torch.cuda.is_available():
            feats_full_data = feats_full_data.cuda()
            # sols_full_data = sols_full_data.cuda()
            objs_full_data = objs_full_data.cuda()
            costs_full_data = costs_full_data.cuda()

            self.weight_model = self.weight_model.cuda()

        early_stopper = EarlyStopper(15, 0.01)

        epoch_times = []

        for epoch in range(epochs):
            start_time = time.perf_counter()
            self.weight_model.train()
            train_loss = 0.0
            opt_sum = 0.0
            for i, data in enumerate(train_loader):
                x, c, y_sol, y_obj, data_feats, data_costs, data_sols, data_objs = data

                if torch.cuda.is_available():
                    x, c, y_obj, data_feats, data_costs, data_objs = x.cuda(), c.cuda() , y_obj.cuda(), data_feats.cuda(), data_costs.cuda(), data_objs.cuda()
                # forward pass
                weights = self._get_weights(x, data_feats)             # [B, N]
                if loss_type == LossType.SFGE:
                    loss = self._sfge_loss(weights, c, y_obj, data_sols, S)
                elif loss_type == LossType.SPO:
                    loss = self._spo_loss(spo_plus, weights, data_costs, c, y_sol, y_obj)
                elif loss_type == LossType.DER:
                    loss = self._der_loss(weights, c, y_obj, data_sols)
                else:
                    raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.DER.")
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
                        x, c, y_obj, data_feats, data_costs, data_objs = x.cuda(), c.cuda(), y_obj.cuda(), data_feats.cuda(), data_costs.cuda(), data_objs.cuda()

                    feats_batch = feats_full_data.unsqueeze(0).expand(len(x), -1, -1).contiguous()  # [B, N, D]
                    costs_batch = costs_full_data.unsqueeze(0).expand(len(x), -1, -1).contiguous()
                    sols = expand_sols(sols_full_data, len(x))  # [B, N, D], tensor or list

                    if data_feats.dim() == 2:
                        data_sols = data_sols.unsqueeze(0)
                        data_feats = data_feats.unsqueeze(0)
                        data_costs = data_costs.unsqueeze(0)

                    sols = cat_sols(data_sols, sols)
                    feats_batch = torch.cat((data_feats, feats_batch), dim=1)
                    costs_batch = torch.cat((data_costs, costs_batch), dim=1)
                        
                    weights = self._get_weights(x, feats_batch)

                    if calc_regret:
                        regret = self._calculate_regret(weights, c, y_obj, sols)
                        opt_sum += np.sum(abs(y_obj.squeeze().cpu().numpy()))
                        regret_loss += np.sum(regret).item()

                    if loss_type == LossType.SFGE:
                        val_loss += self._sfge_loss(weights, c, y_obj, sols, S).item()
                    elif loss_type == LossType.SPO:
                        val_loss += self._spo_loss(spo_plus, weights, costs_batch, c, y_sol, y_obj).item()
                    elif loss_type == LossType.DER:
                        val_loss += self._der_loss(weights, c, y_obj, sols).item()
                    else:
                        raise ValueError("Invalid loss type. Must be LossType.SFGE, LossType.SPO, or LossType.DER.")

                if calc_regret:
                    regret_loss = regret_loss / opt_sum
                val_loss = val_loss / len(val_loader)
                epoch_times.append(time.perf_counter() - start_time)
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
        info = {"mean_epoch_time": np.mean(epoch_times),
                "final_epoch": epoch+1}

        return val_loss, info


# Helper to expand a "sols" structure (tensor or list-of-dicts) like unsqueeze(0).expand(B, ...)
def expand_sols(sols_data, batch_size):
    """Handles both torch tensors and nested lists/arrays containing dicts."""
    if isinstance(sols_data, torch.Tensor):
        return sols_data.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    else:
        # sols_data is assumed to be shape [N, D] as a list/np.array of dicts
        # Result should be [B, N, D]
        if isinstance(sols_data, np.ndarray):
            sols_data = sols_data.tolist()
        return [sols_data for _ in range(batch_size)]  # [B, N, D] as list

def cat_sols(data_sols, sols_batch):
    if isinstance(sols_batch, torch.Tensor):
        if isinstance(data_sols, torch.Tensor) and data_sols.dim() == 2:
            data_sols = data_sols.unsqueeze(0)
        return torch.cat((data_sols, sols_batch), dim=1)
    else:
        if isinstance(data_sols, torch.Tensor):
            data_sols = data_sols.tolist()

        # Both are [B, N, dict]; concat along N for each batch element
        return [
            list(d) + list(s)
            for d, s in zip(data_sols, sols_batch)
        ]  # [B, N_data+N_sols, dict]
    
class GroupedNeuralPrediction(NeuralPrediction):
    def __init__(self, feats, costs, model, weight_model, verbose = False, seed=None):
        super().__init__(feats, costs, model, weight_model, verbose, seed)

        processes = 0
        self.processes = mp.cpu_count() if processes == 0 else processes
        self.pool = ProcessingPool(self.processes)


    def _spo_loss(self, spo_plus, weights, costs_batch, true_costs, true_sols, true_objs) -> torch.Tensor:
        y_hat = torch.einsum('bxn,bn->bx', weights, costs_batch)
        y_hat = self.model.transform_prediction(y_hat)
        true_costs = self.model.transform_prediction(true_costs)
        return spo_plus(y_hat, true_costs, true_sols, true_objs)   

    def train_model(self, epochs=100, batch_size=32, lr=1e-3, val_split=0.11, calc_regret : bool = False, loss_type : LossType = LossType.SFGE, grouped=True):
        g = torch.Generator()
        if self.seed is not None:
            g = g.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        X_train, X_val, y_train, y_val = train_test_split(
            self.features_unadjusted, self.costs_unadjusted, test_size=val_split, random_state=self.seed
        )

        optimizer = optim.Adam(self.weight_model.parameters(), lr=lr)

        spo_plus = SPOPlus(self.model, processes=self.processes)

        X_train = X_train.reshape(-1, X_train.shape[-1])
        y_train = y_train.reshape(-1)
        X_val = X_val.reshape(-1, X_val.shape[-1])
        y_val = y_val.reshape(-1)

        train_loader = torch.utils.data.DataLoader(
            optDatasetSharedPP(self.model, X_train, y_train, self.features_unadjusted.shape[1]),
            batch_size=batch_size, shuffle=True, generator=g
        )
        val_loader = torch.utils.data.DataLoader(
            optDatasetSharedPP(self.model, X_val, y_val, self.features_unadjusted.shape[1]),
            batch_size=batch_size, shuffle=False
        )

        feats_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[0])
        costs_full_data = torch.FloatTensor(train_loader.dataset.get_features_costs()[1])

        if torch.cuda.is_available():
            feats_full_data = feats_full_data.cuda()
            costs_full_data = costs_full_data.cuda()

            self.weight_model = self.weight_model.cuda()

        early_stopper = EarlyStopper(15, 0.01)

        epoch_times = []

        for epoch in range(epochs):
            start_time = time.perf_counter()
            self.weight_model.train()
            train_loss = 0.0
            opt_sum = 0.0
            for i, data in enumerate(train_loader):
                x, c, data_feats, data_costs = data
                y_sol, y_obj = solve_in_pass(c, self.model, self.processes, self.pool)

                if torch.cuda.is_available():
                    x, c, y_sol, y_obj, data_feats, data_costs = x.cuda(), c.cuda(), y_sol.cuda(), y_obj.cuda(), data_feats.cuda(), data_costs.cuda()
                # forward pass
                weights = self._get_weights(x, data_feats)             # [B, X, N]
                if loss_type == LossType.SPO:
                    loss = self._spo_loss(spo_plus, weights, data_costs, c, y_sol, y_obj)
                else:
                    raise ValueError("Invalid loss type. Must be LossType.SPO for grouped prediction.")
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
                    x, c, data_feats, data_costs = data

                    y_sol, y_obj = solve_in_pass(c, self.model, self.processes, self.pool)
                    
                    if torch.cuda.is_available():
                        x, c, y_sol, y_obj, data_feats, data_costs = x.cuda(), c.cuda(), y_sol.cuda(), y_obj.cuda(), data_feats.cuda(), data_costs.cuda()

                    feats_batch = feats_full_data.unsqueeze(0).expand(len(x), -1, -1).contiguous()  # [B, N, D]
                    costs_batch = costs_full_data.unsqueeze(0).expand(len(x), -1).contiguous() 

                    feats_batch = torch.cat((data_feats, feats_batch), dim=1)
                    costs_batch = torch.cat((data_costs, costs_batch), dim=1)
                        
                    weights = self._get_weights(x, feats_batch)

                    if loss_type == LossType.SPO:
                        val_loss += self._spo_loss(spo_plus, weights, costs_batch, c, y_sol, y_obj).item()
                    else:
                        raise ValueError("Invalid loss type. Must be LossType.SPO for grouped prediction.")

                val_loss = val_loss / len(val_loader)

                epoch_times.append(time.perf_counter() - start_time)

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

        info = {"mean_epoch_time": np.mean(epoch_times),
                "final_epoch": epoch+1}

        return val_loss, info



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
        

def custom_collate_fn(batch):
    transposed = list(zip(*batch))
    
    x = default_collate(transposed[0])
    c = default_collate(transposed[1])
    
    sols_index = list(transposed[2])
    
    objs_index = default_collate(transposed[3])
    x_rest = default_collate(transposed[4])
    c_rest = default_collate(transposed[5])
    
    sols_mask = list(transposed[6])
    
    objs_mask = default_collate(transposed[7])
    
    return x, c, sols_index, objs_index, x_rest, c_rest, sols_mask, objs_mask