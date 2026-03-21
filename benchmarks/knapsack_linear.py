import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from torch import nn
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import KernelPrescription, LossType

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
        x: [B, D] query features
        features: [N, D] reference features
        returns: [B, N] normalized weights
        """
        B, N, D = features.shape

        # expand to compare every query with all reference features
        x_exp = x.unsqueeze(1).expand(-1, N, -1)        # [B, N, D]

        # concatenate query with corresponding reference features
        inp = torch.cat([x_exp, features], dim=-1)      # [B, N, 2D]

        weights = self.net(inp).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return weights


# optimization model
class knapSackModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()

    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # model sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
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

def knapsack_generator_factory(num_feat=5, num_item=10):
    def generator(num_data):
        weights, x, c = pyepo.data.knapsack.genData(
            num_data, num_feat, num_item, dim=3, deg=4, noise_width=0.5, seed=135
        )
        optmodel = knapSackModel(weights)
        return x, c, optmodel
    return generator

if __name__ == "__main__":
    sizes = np.linspace(10, 350, 15).astype(int)
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=knapsack_generator_factory(),
        num_runs=10
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
    pipeline.add_model('Nearest Neighbor', WeightingTypeFunction.NEAREST_NEIGBHOUR, param_grid = k_param_grid)
    pipeline.add_model('LOESS', WeightingTypeFunction.LOESS, param_grid = k_param_grid)
    pipeline.add_model('Kernel', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model('Recursive Kernel', WeightingTypeFunction.RKERNEL, param_grid = kernel_param_grid)
    pipeline.add_model('Random Forest', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model('CART', WeightingTypeFunction.CART)
    pipeline.add_model('SAA', WeightingTypeFunction.SAA)
    # pipeline.add_model('Neural Network SFGE',  WeightingTypeFunction.NEURAL, loss=pyepo.predictive.neural.LossType.SFGE, epochs=1000, weight_model = WeightModel)
    pipeline.add_model('Neural Network NOVEL',  WeightingTypeFunction.NEURAL, loss=LossType.NOVEL, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel)
    pipeline.add_model('Neural Network SPO', WeightingTypeFunction.NEURAL, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel)

    # Run and plot
    pipeline.execute()
    pipeline.plot_results('results/knapsack_linear_regret.png', 'Knapsack Benchmark Regret')
    pipeline.plot_normalized_bar_chart(sizes[7], 'Nearest Neighbor', 'results/test.png', 'Knapsack Benchmark Barchart')