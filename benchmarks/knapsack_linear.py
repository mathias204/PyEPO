import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pyepo.data.knapsack import genData
from pyepo.model.grb import optGrbModel
from sklearn.model_selection import train_test_split
import torch
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, weight_model_param_grid, train_param_grid, dfl_model_param_grid, rf_param_grid

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
        x = m.addMVar(shape=(self.num_item,), name="x", vtype=GRB.BINARY)
        # model sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 20)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 20)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 20)
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

def knapsack_generator_factory(num_feat=5, num_item=32, degree=4):
    def generator(num_data, seed):
        weights, x, c = genData(
            num_data, num_feat, num_item, dim=3, deg=degree, noise_width=0.5, seed=seed
        )

        x_train, x_tmp, c_train, c_tmp = train_test_split(
            x, c, test_size=0.2, random_state=seed 
        )

        x_val, x_test, c_val, c_test = train_test_split(
            x_tmp, c_tmp, test_size=0.5, random_state=seed
        )

        optmodel = knapSackModel(weights)
        return x_train, c_train, x_val, c_val, x_test, c_test, optmodel, {}
    return generator

if __name__ == "__main__":
    gp.setParam("OutputFlag", 0)

    degrees = [1,2,4,6,8]
    sizes = np.linspace(500, 500, 1).astype(int)

    for degree in degrees: 
        
        pipeline = PredictOptimizePipeline(
            data_sizes=sizes, 
            data_generator=knapsack_generator_factory(degree=degree),
            num_runs=5
        )

        # Register models to benchmark
        pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
        pipeline.add_model(r'$\hat{z}^{LOESS}_N(x)$', WeightingTypeFunction.LOESS, param_grid = kernel_param_grid)
        pipeline.add_model(r'$\hat{z}^{KR}_N(x)$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
        pipeline.add_model(r'$\hat{z}^{Rec.-KR}_N(x)$', WeightingTypeFunction.RKERNEL, param_grid = kernel_param_grid)
        pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
        pipeline.add_model(r'$\hat{z}^{SPO+}_N(x)$', WeightingTypeFunction.NEURAL, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid,)
        pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER,      weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid)

        pipeline.add_model(r'$z^{SPO+}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SPO, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
        pipeline.add_model(r'$z^{SFGE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
        pipeline.add_model(r'$z^{MSE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.MSE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)

        # Run and plot
        pipeline.execute(save_dir=f"saved_models/knapsack/yet/degree_{degree}/", force_run=False)
        pipeline.save_results_to_csv(f"results/knapsack_normal/attention/degree_{degree}/optimize_results.csv")
        pipeline.plot_weight_distribution(save_dir=f"saved_models/knapsack/yet/degree_{degree}/", data_size=sizes[0], save_path=f"results/knapsack_normal/degree_{degree}/weight_distribution.png", title=f"Weight Distribution for Knapsack (Degree {degree})")
        pipeline.plot_boxplot(sizes[0], f'results/knapsack_normal/attention/degree_{degree}/regret_boxplot.png', 'Knapsack Benchmark Boxplot')

