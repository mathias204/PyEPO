"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PredOpt/predopt-benchmarks
"""
from gurobipy import GRB
from sklearn.model_selection import train_test_split
from pyepo.model.grb import optGrbModel
import gurobipy as gp
import numpy as np
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType
import torch
from pyepo.data.matching import get_cora
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, rf_param_grid, weight_model_param_grid, train_param_grid, dfl_model_param_grid

# Define diversity parameter sets for different instances
params_dict = { 
    1: {'p': 0.1, 'q': 0.1}, 
    2: {'p': 0.25, 'q': 0.25},
    3: {'p': 0.5, 'q': 0.5}  
}
class BipartiteMatching(optGrbModel):
    def __init__(self, p=0.25, q=0.25, relaxation=True) -> None:
        self.p, self.q = p, q
        self.relaxation = relaxation

        super().__init__() 

    def _getModel(self):
        # create model
        model = gp.Model("BipartiteMatching")
        
        # decision variables mapped to 1D
        x = model.addMVar(
            shape=(2500,), 
            lb=0, 
            ub=1, 
            vtype=GRB.CONTINUOUS if self.relaxation else GRB.BINARY, 
            name="x"
        )

        model.modelSense = GRB.MAXIMIZE
        
        # constraints not depending on the specific instance
        for i in range(50):
            model.addConstr(gp.quicksum(x[i * 50 + j] for j in range(50)) <= 1)
        for j in range(50):
            model.addConstr(gp.quicksum(x[i * 50 + j] for i in range(50)) <= 1)

        return model, x
    
    def setM(self, M):
        # ensure M is a flattened 1D array
        M = M.reshape(2500,)

        total_matches = self.x.sum()

        # Constraint 1: sum(phi_ij * x_ij) >= rho1 * sum(x_ij)
        # using matrix multiplication (@) for dot product of 1D arrays
        c1 = self._model.addConstr(
            (M @ self.x) >= self.p * total_matches, 
            name="phi_constraint_1"
        )
        
        # Constraint 2: sum((1 - phi_ij) * x_ij) >= rho2 * sum(x_ij)
        c2 = self._model.addConstr(
            ((1 - M) @ self.x) >= self.q * total_matches, 
            name="phi_constraint_2"
        )
        
        self._model.update()
        
        return [c1, c2]
    
    def removeM(self, constraints):
        for c in constraints:
            self._model.remove(c)
        self._model.update()

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
            
        c = c.reshape(-1)
            
        # Case 1: x shape (2500,)
        if x.ndim == 1:
            return np.dot(c, x)

        # Case 2: x shape (B, 2500)
        elif x.ndim == 2:
            return np.sum(c * x, axis=-1)

        else:
            raise ValueError(f"Unsupported x shape {x.shape}")
        
    def transform_prediction(self, y_pred):
        """
        Transforms predicted energy prices into objective coefficients for z_{jit}.

        Args:
            y_pred (torch.Tensor): Predicted energy prices
            
        Returns:
            torch.Tensor: Flattened cost coefficients matching z_{jit} extraction order
        """
        y_pred = y_pred * 1000 # scale costs for numerical stability
        return y_pred.reshape(y_pred.shape[0], 2500)

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray / list): cost of objective function
        """
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)
        
        c = c * 1000 # scale costs for numerical stability
        c = c.reshape(2500,)

        self._model.setObjective(self._objective_fun(c))

    def setWeightObj(self, W, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            W (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        if c.shape[0] != W.shape[1]:
            raise ValueError("Weights and costs must have same first dimension.")
        
        obj_coefficients = np.dot(W, c)

        obj_coefficients = obj_coefficients * 1000 # scale costs for numerical stability
        obj_coefficients = obj_coefficients.reshape(2500,)

        self._model.setObjective(self._objective_fun(obj_coefficients))

def matching_generator_factory(instance = 1):
    def generator(groups, seed): 
        x, y , m = get_cora()

        x = x[:groups]
        y = y[:groups]
        m = m[:groups]

        x_train, x_tmp, y_train, y_tmp, m_train, m_tmp = train_test_split(
            x, y, m, test_size=2, random_state=seed 
        )

        x_test, x_val, y_test, y_val, m_test, m_val = train_test_split(
            x_tmp, y_tmp, m_tmp, test_size=1, random_state=seed 
        )

        optmodel = BipartiteMatching(params_dict[instance]["p"], params_dict[instance]["q"], relaxation=True)

        # Pack standard variables normally, put extras in a dict
        aux_data = {
            'train': m_train,
            'val': m_val,
            'test': m_test
        }

        return x_train, y_train, x_val, y_val, x_test, y_test, optmodel, aux_data
    return generator




if __name__ == "__main__":
    gp.setParam("OutputFlag", 0)
    instances = [1, 2, 3]
    sizes = np.linspace(5, 5, 1).astype(int)
    
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
    for instance in instances:

        pipeline = PredictOptimizePipeline(
            data_sizes=sizes, 
            data_generator=matching_generator_factory(),
            num_runs=5
        )
        # Register models to benchmark
        pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
        pipeline.add_model(r'$\hat{z}^{LOESS}_N(x)$', WeightingTypeFunction.LOESS, param_grid = kernel_param_grid)
        pipeline.add_model(r'$\hat{z}^{KR}_N(x)$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
        pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
        pipeline.add_model(r'$\hat{z}^{SPO+}_N(x)$', WeightingTypeFunction.NEURAL_GROUPED, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid)

        pipeline.add_model(r'$z^{SPO+}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SPO, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
        pipeline.add_model(r'$z^{SFGE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
        pipeline.add_model(r'$z^{MSE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.MSE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
        
        # Run and plot
        pipeline.execute(save_dir=f"saved_models/matching/instance_{instance}/", force_run=True)
        pipeline.save_results_to_csv(f"results/matching/instance_{instance}/results.csv")
        pipeline.plot_boxplot(sizes[0], f'results/matching/instance_{instance}/bipartite_boxplot.png', 'Shortest Path Benchmark Boxplot')
