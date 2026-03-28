import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import KernelPrescription, LossType
from pyepo.data.generate_california_house_price_mapping import generate_california_house_prices_mapping
from sklearn.preprocessing import StandardScaler

# Weight model
class WeightModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features): 
        """
        x: [B, X, D] query features
        features: [B, N, D] reference features
        returns: [B, X, N] normalized weights
        """
        B, N, D = features.shape
        _, X, _ = x.shape

        # expand to compare every query with all reference features
        x_exp = x.unsqueeze(2).expand(-1, -1, N, -1) # [B, X, N, D]

        feat_exp = features.unsqueeze(1).expand(-1, X, -1, -1)

        # concatenate query with corresponding reference features
        inp = torch.cat([x_exp, feat_exp], dim=-1)

        weights = self.net(inp).squeeze(-1)

        weights = torch.softmax(weights, dim=-1)
        return weights


# optimization model
class knapSackModel(optGrbModel):
    def __init__(self, weights, capacities):
        self.weights = np.array(weights)
        self.capacities = np.array(capacities)
        self.num_item = self.weights.shape[1]
        super().__init__()

    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # model sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(self.weights.shape[0]):
            m.addConstr(gp.quicksum(self.weights[i][j] * x[j] for j in range(self.num_item)) <= self.capacities[i],
                        name=f"capacity_{i}")
        return m, x
    
    def cal_obj(self, c, x):
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        # check if x is a PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)
        if c.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mismatch: c has {c.shape[-1]} features, expected {x.shape[-1]}.")
            
        # Case 1: x shape (C)
        if x.ndim == 1:
            # c shape (C,)
            return np.dot(c, x)

        # Case 2: x shape (B, C)
        elif x.ndim == 2:
            # c can be (C,) or (B, C)
            if c.ndim == 1:
                # simple broadcasting
                return np.sum(c * x, axis=-1)
            elif c.ndim == 2:
                return np.sum(c * x, axis=-1)
            else:
                raise ValueError("c must be shape (C,) or (B, C)")

        # Case 3: x shape (B, S, C)
        elif x.ndim == 3:
            # c can be (C,) or (B, C)
            if c.ndim == 1:
                c_exp = c[None, None, :]       # broadcast to (1, 1, C)
            elif c.ndim == 2:
                c_exp = c[:, None, :]          # broadcast to (B, 1, C)
            else:
                raise ValueError("c must be shape (C,) or (B, C)")

            return np.sum(c_exp * x, axis=-1)   # output (B, S)

        else:
            raise ValueError(f"Unsupported x shape {x.shape}")
    
    def setWeightObj(self, W, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            W (np.ndarray): shape (N,), weights for each sample
            C (np.ndarray): shape (N, C), cost vectors for each sample
        """
        # if c.shape[1] != self.num_cost:
        #     raise ValueError("Cost vector dimension mismatch.")
        if c.shape[0] != W.shape[1]:
            raise ValueError("Weights and costs must have same first dimension.")
        
        obj_coefficients = np.dot(W, c)

        self._model.setObjective(self._objective_fun(obj_coefficients))


def knapsack_generator_factory(dims=3, num_item=20):
    def generator(num_data):
        x, c = generate_california_house_prices_mapping(num_data, num_item)

        x_tmp, x_test, c_tmp, c_test = train_test_split(
            x, c, test_size=0.1, random_state=0 
        )

        x_train, x_val, c_train, c_val = train_test_split(
            x_tmp, c_tmp, test_size=0.11, random_state=0 
        )

        s_scaler = StandardScaler()
        train_shape = x_train.shape
        val_shape = x_val.shape
        test_shape = x_test.shape

        # Reshape to 2D: (samples * timesteps, features)
        x_train_2d = x_train.reshape(-1, train_shape[-1])
        x_val_2d = x_val.reshape(-1, val_shape[-1])
        x_test_2d = x_test.reshape(-1, test_shape[-1])

        # Fit and transform on 2D data
        x_train_scaled = s_scaler.fit_transform(x_train_2d.astype(np.float64))
        x_val_scaled = s_scaler.transform(x_val_2d.astype(np.float64))
        x_test_scaled = s_scaler.transform(x_test_2d.astype(np.float64))

        # Reshape back to 3D
        x_train = x_train_scaled.reshape(train_shape)
        x_val = x_val_scaled.reshape(val_shape)
        x_test = x_test_scaled.reshape(test_shape)

        weights = np.random.randint(1, 10, size=(dims, num_item))
        capacities = np.array(0.1 * np.sum(weights, axis=1))

        optmodel = knapSackModel(weights, capacities)

        return x_train, c_train, x_val, c_val, x_test, c_test, optmodel
    return generator

if __name__ == "__main__":
    sizes = np.linspace(4, 200, 20).astype(int)
    # sizes = np.linspace(200, 200, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=knapsack_generator_factory(),
        num_runs=5
    )

    k_param_grid = {
        "k": [1, 3, 5, 10],
    }
    kernel_param_grid = {
        **k_param_grid,
        "kernel" : [
            KernelPrescription._naive_kernel,
            KernelPrescription._epanechnikov_kernel,
            KernelPrescription._tricubic_kernel,
        ]
    }

    rf_param_grid = {
        "n_est": [50, 100, 200],
        "depth": [5, 10, 20, None],
    }

    weight_model_param_grid = {
        "hidden_dim": [32, 64, 128],
        "dropout": [0, 0.1]
    }

    train_param_grid = {
        "epochs": [1000],
        "batch_size": [32, 64],
        "lr": [1e-3, 5e-4],
    }

    # Register models to benchmark
    pipeline.add_model('Nearest Neighbor', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model('LOESS', WeightingTypeFunction.LOESS, param_grid = k_param_grid)
    # pipeline.add_model('Kernel', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model('Recursive Kernel', WeightingTypeFunction.RKERNEL, param_grid = kernel_param_grid)
    pipeline.add_model('Random Forest', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    # pipeline.add_model('CART', WeightingTypeFunction.CART)
    # pipeline.add_model('SAA', WeightingTypeFunction.SAA)
    # pipeline.add_model('Neural Network SFGE',  WeightingTypeFunction.NEURAL, loss=pyepo.predictive.neural.LossType.SFGE, epochs=1000, weight_model = WeightModel)
    # pipeline.add_model('Neural Network NOVEL',  WeightingTypeFunction.NEURAL, loss=LossType.NOVEL, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel)
    pipeline.add_model('Neural Network SPO', WeightingTypeFunction.NEURAL, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel)

    # Run and plot
    pipeline.execute()
    pipeline.plot_results('results/knapsack_houses_regret.png', 'Knapsack Houses Benchmark Regret')
    # pipeline.plot_normalized_bar_chart(sizes[0], 'Nearest Neighbor', 'results/test.png', 'Knapsack Benchmark Barchart')