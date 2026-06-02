"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/ML-KULeuven/Solver-Free-DFL/
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from pyepo.model.grb import optGrbModel
from sklearn.model_selection import train_test_split
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType
from pyepo.data.generate_california_house_price_mapping import generate_california_house_prices_mapping
from sklearn.preprocessing import StandardScaler
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, rf_param_grid, weight_model_param_grid, train_param_grid, dfl_model_param_grid

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
        
        y_hat = np.dot(W, c)
        self.setObj(y_hat)


def knapsack_generator_factory(dims=4, num_item=25):
    def generator(num_data, seed=42):
        x, c = generate_california_house_prices_mapping(num_data, num_item, seed=seed)

        x_train, x_tmp, c_train, c_tmp = train_test_split(
            x, c, test_size=0.2, random_state=seed 
        )

        x_val, x_test, c_val, c_test = train_test_split(
            x_tmp, c_tmp, test_size=0.5, random_state=seed
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
        capacities = np.array(0.8 * np.sum(weights, axis=1))

        optmodel = knapSackModel(weights, capacities)

        return x_train, c_train, x_val, c_val, x_test, c_test, optmodel, {}
    return generator

if __name__ == "__main__":
    sizes = np.linspace(50, 50, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=knapsack_generator_factory(),
        num_runs=5
    )

    train_param_grid = {
        **train_param_grid,
        "grouped": [True],
    }

    weight_model_param_grid = {
        **weight_model_param_grid,
        "shared": [True],
    }

    dfl_model_param_grid = {
        **dfl_model_param_grid,
        "shared": [True]
    }

    # Register models to benchmark
    pipeline.add_model(r'$\hat{z}^{kNN}_N$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{LOESS}_N$', WeightingTypeFunction.LOESS, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{KR}_N$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{RF}_N$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model(r'$\hat{z}^{SPO+}_N$', WeightingTypeFunction.NEURAL_GROUPED, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid)

    pipeline.add_model(r'$z^{SPO+}$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SPO, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
    pipeline.add_model(r'$z^{SFGE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
    pipeline.add_model(r'$z^{MSE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.MSE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)


    # Run and plot
    pipeline.execute(save_dir="saved_models/houses/", force_run=True)
    pipeline.save_results_to_csv('results/houses/knapsack_houses_results.csv')
    pipeline.plot_boxplot(sizes[0], 'results/houses/knapsack_houses_boxplot.png', 'Knapsack Houses Benchmark Boxplot')