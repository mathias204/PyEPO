import numpy as np
import pyepo
from pyepo.model.omo import optOmoModel
# from pyepo.model.grb import optGrbModel
import pyomo.environ as pyo
import torch
from torch import nn
from pyepo import EPO
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType, KernelPrescription

class WeightModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(dropout),
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
class portfolioModel(optOmoModel):
    def __init__(self, n_assets, beta):
        self.n_assets = n_assets
        self.beta = beta
        self.modelSense = EPO.MAXIMIZE
        super().__init__(solver="ipopt")

    def _getModel(self):
        # ceate a model
        model = pyo.ConcreteModel("Portfolio")

        model.I = pyo.RangeSet(0, self.n_assets)
        model.x = pyo.Var(model.I, bounds=(0.0, 1.0))

        model.budget = pyo.Constraint(expr = sum(model.x[i] for i in model.I) == 1)

        return model, model.x
    
    def _objective_fun(self, c):
        """
        Logarithmic objective function

        Args:
            c (np.ndarray): cost vector for one sample
            
        """
        return pyo.log(1 + self.beta * self.x[0] + pyo.quicksum(c[i] * self.x[i + 1] for i in range(len(c))))
    
    def cal_obj(self, c, x):
        """
        Compute objective value for given cost and decision vector.

        Args:
            c (ndarray or Tensor): cost vector(s), shape (C) or (B, C)
            x (ndarray or Tensor): decision variables, shape (C + 1), (B, C + 1), or (B, S, C + 1)
        Returns:
            ndarray: objective values with shape matching x excluding the last dimension.
        """
        # Convert to numpy if tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)

        C = c.shape[-1]
        # Check dimension compatibility
        if c.shape[-1] != x.shape[-1] - 1:
            raise ValueError(f"Mismatch: c has {c.shape[-1]} features, expected {x.shape[-1] - 1}.")

        # Determine dimensionality of x
        # Always separate x0 from the features x1...xC
        x0 = x[..., 0]
        x_feat = x[..., 1:]

        if x_feat.ndim == 1:
            # x_feat shape: (C)
            # c can be (C) or (1, C) or (B, C) but broadcast to (C) or (1, C) does not work
            # So only allow c shape (C) in this case
            if c.ndim != 1:
                raise ValueError("For x shape (C plus one) c must be shape (C)")
            c_b = c
        elif x_feat.ndim == 2:
            # x_feat shape: (B, C)
            B = x_feat.shape[0]
            if c.ndim == 1:
                # shared c across batch
                c_b = c.reshape(1, C)
            elif c.ndim == 2:
                # per batch c
                if c.shape[0] != B:
                    raise ValueError("Batch size of c does not match x")
                c_b = c
            else:
                raise ValueError("Invalid c shape")
        elif x_feat.ndim == 3:
            # x_feat shape: (B, S, C)
            B = x_feat.shape[0]
            S = x_feat.shape[1]
            if c.ndim == 1:
                # shared c across batch and scenarios
                c_b = c.reshape(1, 1, C)
            elif c.ndim == 2:
                # per batch c, shared across scenarios
                if c.shape[0] != B:
                    raise ValueError("Batch size of c does not match x")
                c_b = c.reshape(B, 1, C)
            else:
                raise ValueError("Invalid c shape")
        else:
            raise ValueError("Unsupported x shape")

        # Compute dot product
        dot = np.sum(c_b * x_feat, axis = x_feat.ndim - 1)

        # Compute objective
        obj = np.log(1 + self.beta * x0 + dot)

        return obj
    
def portfolio_generator_factory(m=50, p = 4, deg=4, e=1):
    optmodel = portfolioModel(m, 0.08) 
    def generator(num_data):
        _, x, c = pyepo.data.portfolio.genData(num_data=num_data, num_features=p, num_assets=m, deg=deg, noise_level=e, seed=42)
        return x, c,optmodel
    return generator
    
if __name__ == "__main__":
    sizes = np.linspace(10, 200, 5).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=portfolio_generator_factory(),
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
    pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{LOESS}_N(x)$', WeightingTypeFunction.LOESS, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{KR}_N(x)$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{Rec.-KR}_N(x)$', WeightingTypeFunction.RKERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model(r'$\hat{z}^{CART}_N(x)$', WeightingTypeFunction.CART)
    pipeline.add_model(r'$\hat{z}^{SAA}_N(x)$', WeightingTypeFunction.SAA)
    # pipeline.add_model('Neural Network SFGE',  WeightingTypeFunction.NEURAL, loss=pyepo.predictive.neural.LossType.SFGE, epochs=1000, weight_model = WeightModel)
    pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER,      weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel) # Discrete Expectation Regret


    # Run and plot
    pipeline.execute()
    pipeline.plot_results('results/portfolio_regret.png', 'Portfolio Benchmark Regret')
    pipeline.plot_boxplot(sizes[2],'results/portfolio_boxplot.png', "Portfolio Benchmark Boxplot")
    pipeline.plot_weight_distribution(150, 'results/portfolio_weights.png', 'Portfolio weight distribution')
