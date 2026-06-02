"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PredOpt/predopt-benchmarks
"""
from gurobipy import GRB
from pyepo.model.grb import optGrbModel
import gurobipy as gp
import numpy as np
from pyepo.data.energy import get_data, get_instance_config
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType
import torch
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, rf_param_grid, weight_model_param_grid, dfl_model_param_grid, train_param_grid

class SolveICON(optGrbModel):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds
    def __init__(self,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
        relax=True,
        verbose=False,method=-1,**h):
        self.nbMachines  = nbMachines
        self.nbTasks = nbTasks
        self.nbResources = nbResources
        self.MC = MC
        self.U =  U
        self.D = D
        self.E = E
        self.L = L
        self.P = P
        self.idle = idle
        self.up = up
        self.down = down
        self.q= q
        self.relax = relax
        self.verbose = verbose
        self.method = method

        super().__init__()
        
    def _getModel(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U = self.U
        D = self.D
        E = self.E
        L = self.L
        q = self.q
        N = 1440 // q
        self.N = N

        V = self.nbTasks * self.nbMachines * N

        M = gp.Model("icon")
        M.modelSense = GRB.MINIMIZE

        if not self.verbose:
            M.setParam('OutputFlag', 0)

        vtype = GRB.CONTINUOUS if self.relax else GRB.BINARY
        
        # Flat decision variable
        x = M.addVars(V, lb=0., ub=1., vtype=vtype, name="x")

        # Helper function to map 3D coordinates to the flat 1D index
        def flat_idx(f, m, t):
            return f * (self.nbMachines * N) + m * N + t

        # Adjusted constraints using the flat index
        for f in Tasks:
            M.addConstr(gp.quicksum(x[flat_idx(f, m, t)] for m in Machines for t in range(E[f])) == 0)
            
            start_limit = L[f] - D[f] + 1
            M.addConstr(gp.quicksum(x[flat_idx(f, m, t)] for m in Machines for t in range(start_limit, N)) == 0)
            
            M.addConstr(gp.quicksum(x[flat_idx(f, m, t)] for m in Machines for t in range(N)) == 1)

        # Capacity requirement constraints
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr(
                        gp.quicksum(
                            x[flat_idx(f, m, t1)] * U[f][r]
                            for f in Tasks
                            for t1 in range(max(0, t - D[f] + 1), t + 1)
                        ) <= MC[m][r]
                    )

        M.update()
        self.model = M
        self.x = x

        return M, x

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

        if c.shape[-1] != len(self.x):
            c = self.transform_prediction(c)
        self._model.setObjective(self._objective_fun(c))

    def cal_obj(self, price, x):
        """
        Calculates the objective value for the scheduling problem.
        Supports price as (N,) or (B, N) and x as flat arrays or multi-dimensional grids.
        """
        # 1. Convert price to a 2D PyTorch Tensor (B, N) for transform_prediction
        if not isinstance(price, torch.Tensor):
            price_tensor = torch.tensor(price, dtype=torch.float32)
        else:
            price_tensor = price.clone().detach()
            
        is_single_batch = False
        if price_tensor.ndim == 1:
            price_tensor = price_tensor.unsqueeze(0)
            is_single_batch = True

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            # This safely converts Python lists to NumPy arrays
            x = np.asarray(x, dtype=np.float32)

        # 2. Get the pre-calculated objective coefficients directly from the model
        # Shape will be (B, Tasks * Machines * N)
        c_flattened = self.transform_prediction(price_tensor).detach().cpu().numpy()

        # 3. Standardize x to numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # 5. Calculate the objective via dot product
        if x.ndim == 1 and c_flattened.shape[0] == 1:
            # Single instance, single cost (1D dot product)
            total_cost = np.dot(c_flattened[0], x)
            return total_cost if not is_single_batch else total_cost.item()
            
        elif x.ndim == 2 and c_flattened.shape[0] == x.shape[0]:
            # Batched instance: element-wise multiplication followed by sum over the variable dimension
            batch_costs = np.sum(c_flattened * x, axis=1)
            return batch_costs
            
        else:
            # Handle edge cases (e.g., batched prices with a single x solution)
            batch_costs = np.sum(c_flattened * x, axis=1)
            return batch_costs
        
    def transform_prediction(self, y_pred):
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

        is_unbatched = False
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            is_unbatched = True
        batch_size = y_pred.shape[0]
        N = self.N
        nbTasks = self.nbTasks
        nbMachines = self.nbMachines
        D = self.D
        P = self.P
        q = self.q

        y_adj_list = []
        for f in range(nbTasks):
            # Construction of the window matrix A for task f
            A_f = torch.zeros((N, N), device=y_pred.device)
            duration = int(D[f])
            phi_f = P[f] * (q / 60.0)

            for t in range(N):
                end_t = min(t + duration, N)
                A_f[t, t:end_t] = 1.0

            # Matrix-vector multiplication equivalent to the moving window sum
            y_f = phi_f * torch.matmul(y_pred, A_f.T)
            
            # Expand across the machine dimension
            y_f_expanded = y_f.unsqueeze(1).expand(-1, nbMachines, -1)
            y_adj_list.append(y_f_expanded)

        # Stack along tasks and flatten to exactly match the 1D extraction order of variables
        y_adjusted = torch.stack(y_adj_list, dim=1)
        y_flattened = y_adjusted.reshape(batch_size, -1)
        return y_flattened.squeeze(0) if is_unbatched else y_flattened
    
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

        self.setObj(obj_coefficients)


def energy_generator_factory(instance = 1):
    def generator(num_groups, seed = 42):
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(num_groups=num_groups, seed=seed)

        params = get_instance_config("data/load{}/day01.txt".format(instance))
        optmodel = SolveICON(**params)

        return x_train, y_train, x_val, y_val, x_test, y_test, optmodel, {}
    return generator


if __name__ == "__main__":
    gp.setParam("OutputFlag", 0)

    num_groups = 50
    sizes = np.linspace(num_groups, num_groups, 1).astype(int)

    instances = [1]

    train_param_grid = {
        **train_param_grid,
        "grouped": [True],
        "batch_size": [64],
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
            data_generator=energy_generator_factory(instance=instance),
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
        pipeline.execute(save_dir=f"saved_models/energy/instance_{instance}/", force_run=True)
        pipeline.save_results_to_csv(f'results/energy/instance_{instance}/energy_results.csv')
        pipeline.plot_boxplot(sizes[0], f'results/energy/instance_{instance}/energy_schedule_boxplot.png', f'Energy-cost aware scheduling - Regret boxplot Instance {instance}')