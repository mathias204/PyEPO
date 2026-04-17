from gurobipy import GRB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyepo.model.grb import optGrbModel
import gurobipy as gp
import numpy as np
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import KernelPrescription, LossType
import torch
from torch import nn
from pyepo.data.matching import get_cora


# Define diversity parameter sets for different instances
params_dict = { 
    1: {'p': 0.1, 'q': 0.1}, 
    2: {'p': 0.25, 'q': 0.25},
    3: {'p': 0.5, 'q': 0.5}  
}



# Weight model
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
        features: [B, N, D] reference features
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




class BipartiteMatching(optGrbModel):
    def __init__(self,m ,p=0.25, q=0.25, relaxation=True) -> None:
        self.p, self.q = p,q
        self.M = m
        self.relaxation = relaxation

        super().__init__() 


    def _getModel(self):
        # create model
        model = gp.Model("BipartiteMatching")
        # decision variables
        x = model.addMVar(shape=(50,50), lb=0, ub=1, vtype=GRB.BINARY if not self.relaxation else GRB.CONTINUOUS, name="x")

        model.modelSense = GRB.MAXIMIZE
        
        # constraints not depending on the specific instance
        for i in range(50):
            model.addConstr(gp.quicksum(x[i, j] for j in range(50)) <= 1)
        for j in range(50):
            model.addConstr(gp.quicksum(x[i, j] for i in range(50)) <= 1)

        return model, x
    
    def setM(self, M):
        M = M.reshape(50,50)

        total_matches = self.x.sum()

        # Constraint 1: sum(phi_ij * x_ij) >= rho1 * sum(x_ij)
        c1 = self._model.addConstr(
            (M * self.x).sum() >= self.p * total_matches, 
            name="phi_constraint_1"
        )
        
        # Constraint 2: sum((1 - phi_ij) * x_ij) >= rho2 * sum(x_ij)
        c2 = self._model.addConstr(
            ((1 - M) * self.x).sum() >= self.q * total_matches, 
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
            
        if c.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mismatch: c has {c.shape[-1]} features, expected {x.shape[-1]}.")
            
        # Case 1: x shape (50, 50)
        if x.ndim == 2:
            # c shape (2500,)
            return np.dot(c, x.flatten())

        # Case 2: x shape (B, 50, 50)
        elif x.ndim == 3:
            # c can be (B, 2500) or (2500,)
            x_flat = x.reshape(x.shape[0], -1) 
            return np.sum(c * x_flat, axis=-1)

        # # Case 3: x shape (B, S, C)
        # elif x.ndim == 3:
        #     # c can be (C,) or (B, C)
        #     if c.ndim == 1:
        #         c_exp = c[None, None, :]       # broadcast to (1, 1, C)
        #     elif c.ndim == 2:
        #         c_exp = c[:, None, :]          # broadcast to (B, 1, C)
        #     else:
        #         raise ValueError("c must be shape (C,) or (B, C)")

        #     return np.sum(c_exp * x, axis=-1)   # output (B, S)

        else:
            raise ValueError(f"Unsupported x shape {x.shape}")

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
        c = c.reshape(50,50)

        self._model.setObjective(self._objective_fun(c))

    def setWeightObj(self, W, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            W (np.ndarray): shape (N,), weights for each sample TODO shape is not correct
            C (np.ndarray): shape (N, C), cost vectors for each sample
        """
        # if c.shape[1] != self.num_cost:
        #     raise ValueError("Cost vector dimension mismatch.")
        if c.shape[0] != W.shape[1]:
            raise ValueError("Weights and costs must have same first dimension.")
        
        obj_coefficients = np.dot(W, c)

        obj_coefficients = obj_coefficients * 1000 # scale costs for numerical stability
        obj_coefficients = obj_coefficients.reshape(50,50)


        self._model.setObjective(self._objective_fun(obj_coefficients))

        print(f"Number of variables: {self._model.NumVars}")
        print(f"Number of constraints: {self._model.NumConstrs}")
        print(f"Number of non-zeros: {self._model.NumNZs}")


def matching_generator_factory(instance = 1):
    def generator(num_data): #TODO: num_data is not yet implemented
        x, y , m = get_cora()

        x_tmp, x_test, y_tmp, y_test, m_tmp, m_test = train_test_split(
            x, y, m, test_size=0.1, random_state=0 
        )

        x_train, x_val, y_train, y_val, m_train, m_val = train_test_split(
            x_tmp, y_tmp, m_tmp, test_size=0.11, random_state=0 
        )

        optmodel = BipartiteMatching(params_dict[instance].values(), relaxation=True)

        # Pack standard variables normally, put extras in a dict
        aux_data = {
            'train': m_train,
            'val': m_val,
            'test': m_test
        }

        return x_train, y_train, x_val, y_val, x_test, y_test, optmodel, aux_data
    return generator




if __name__ == "__main__":
    sizes = np.linspace(10, 350, 15).astype(int)
    sizes = np.linspace(200, 200, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=matching_generator_factory(),
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
    # pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER,      weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel) # Discrete Expectation Regret
    pipeline.add_model(r'$\hat{z}^{SPO+}_N(x)$', WeightingTypeFunction.NEURAL_GROUPED, loss=LossType.SPO, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid, weight_model = WeightModel)

    # Run and plot
    pipeline.execute()
    # pipeline.plot_results('results/shortest_path_linear_regret.png', 'Shortest Path Benchmark Regret')
    # pipeline.plot_normalized_bar_chart(sizes[7], 'Nearest Neighbor', 'results/test.png', 'Shortest Path Benchmark Barchart')
    pipeline.plot_boxplot(sizes[0], 'results/shortest_path_linear_boxplot.png', 'Shortest Path Benchmark Boxplot')
    # pipeline.plot_weight_distribution(150, 'results/shortest_path_weights.png', 'Shortest Path Weight distribution')
