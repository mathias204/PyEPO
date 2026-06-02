"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PyDFLT/PyDFLT
"""
import numpy as np
from gurobipy import GRB
import gurobipy as gp
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive import LossType
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, rf_param_grid, weight_model_param_grid, train_param_grid, dfl_model_param_grid
from pyepo.model.grb import optGrbModel
import torch
from pyepo.data.new_portfolio import portfolio_pre_run_hook

# optimization model
class portfolioModel(optGrbModel):
    def __init__(self, n_assets, beta):
        self.n_assets = n_assets
        self.beta = beta
        super().__init__()

    def _getModel(self):
        # ceate a model
        model = gp.Model()

        model.modelSense = GRB.MAXIMIZE

        x = model.addMVar((self.n_assets,), name="x", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)

        model.addConstr(x.sum() <= 1)
        return model, x
    
    def _build_objective_components(self, c, name_suffix):
        """
        Builds and returns the Gurobi variables and constraints for a cost vector.
        """
        z_aux = self._model.addVar(lb=1e-6, name=f"z_{name_suffix}")
        y_aux = self._model.addVar(name=f"y_{name_suffix}")

        expr = (
            1
            + self.beta * (1 - self.x.sum())
            + c @ self.x
            # + gp.quicksum(c[i] * self.x[i] for i in range(self.n_assets))
        )

        lin_constr = self._model.addConstr(z_aux == expr)
        log_constr = self._model.addGenConstrLog(z_aux, y_aux)

        return z_aux, y_aux, lin_constr, log_constr

    def _clear_transient_components(self):
        """
        Removes all auxiliary variables and constraints from the previous objective.
        """
        self._model.update()  # Ensure model is up-to-date before removal
        if hasattr(self, "_transient_obj_components"):
            for comp in self._transient_obj_components:
                self._model.remove(comp)
        self._transient_obj_components = []

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")
        
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        self._clear_transient_components()
            
        z_aux, y_aux, lin_constr, log_constr = self._build_objective_components(c, "single")
        
        self._transient_obj_components.extend([z_aux, y_aux, lin_constr, log_constr])
            
        self._model.setObjective(y_aux)
    
    def setWeightObj(self, w, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            w (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        if c.shape[1] != self.num_cost:
            raise ValueError("Cost vector dimension mismatch.")
        if c.shape[0] != len(w):
            raise ValueError("Weights and costs must have same first dimension.")
        
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy()
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()

        self._clear_transient_components()
        
        # Localized cache to prevent duplicate components within the current batch
        obj_terms = []
        
        for i in range(len(w)):
            if w[i] <= 1e-6:
                continue 
            z_aux, y_aux, lin_constr, log_constr = self._build_objective_components(c[i], f"sample_{i}")
            self._transient_obj_components.extend([z_aux, y_aux, lin_constr, log_constr])
                
            obj_terms.append(w[i] * y_aux)
        
        obj = gp.quicksum(obj_terms)
        self._model.setObjective(obj)

    def cal_obj(self, c, x):
        """
        Compute objective value for given cost and decision vector.

        Args:
            c (ndarray or Tensor): cost vector(s), shape (C) or (B, C)
            x (ndarray or Tensor): decision variables, shape (C), (B, C), or (B, S, C)
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

        # Check dimensionality compatibility
        if c.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mismatch: c has {c.shape[-1]} features, expected {x.shape[-1]}.")
    

        if x.ndim == 3 and c.ndim ==2:
            c = c[:, None, :]  # reshape c to (B, 1, C) for broadcasting

        # Compute dot product
        dot = np.sum(c * x, axis = -1)

        # Compute objective
        obj = np.log(1 + self.beta * (1 - np.sum(x, axis=-1)) + dot)

        return obj

def portfolio_generator_factory():
    def generator(num_data, seed=5):
        seed = seed + 5
        x_train, c_train, x_val, c_val, x_test, c_test, beta = portfolio_pre_run_hook(seed, train_ratio=0.8)
        optmodel = portfolioModel(n_assets=c_train.shape[1], beta=beta)
        return x_train, c_train, x_val, c_val, x_test, c_test, optmodel, {}
    
    return generator

if __name__ == "__main__":
    gp.setParam("OutputFlag", 0)
    size = 2898
    sizes = np.linspace(size, size, 1).astype(int)

    train_param_grid = {
        **train_param_grid,
        "batch_size": [32],
    }

    weight_model_param_grid = {
        **weight_model_param_grid,
    }

    dfl_model_param_grid = {
        **dfl_model_param_grid,
    }

    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=portfolio_generator_factory(),
        num_runs=5
    )

    # Register models to benchmark
    pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{KR}_N(x)$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER,weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid) # Discrete Expectation Regret

    pipeline.add_model(r'$z^{SFGE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
    pipeline.add_model(r'$z^{MSE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.MSE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)

    # Run and plot
    pipeline.execute(save_dir="saved_models/portfolio/", force_run=True)
    pipeline.save_results_to_csv('results/portfolio/optimize_results.csv')
    pipeline.plot_boxplot(sizes[0], 'results/portfolio/portfolio_boxplot.png', 'Portfolio Benchmark Boxplot')
