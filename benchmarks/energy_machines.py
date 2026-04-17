
from gurobipy import GRB
from sklearn.preprocessing import StandardScaler
from pyepo.model.grb import optGrbModel
import gurobipy as gp
import numpy as np
from pyepo.data.energy import get_data, get_instance_config
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import KernelPrescription, LossType
import torch
from torch import nn

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

    # @property
    # def num_cost(self):
    #     """
    #     number of cost to be predicted
    #     """
    #     return self.nbTasks
       
        
    def _getModel(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        relax = self.relax
        q= self.q
        N = 1440//q

        M = gp.Model("icon")
        if not self.verbose:
            M.setParam('OutputFlag', 0)
        if relax:
            x = M.addVars(Tasks, Machines, range(N), lb=0., ub=1., vtype=GRB.CONTINUOUS, name="x")
        else:
            x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")


        M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
        M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)
        M.addConstrs(( gp.quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

        # capacity requirement
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr( gp.quicksum( gp.quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])   
        # M = M.presolve()
        M.update()
        self.model = M

        # self.x = dict()
        # for var in M.getVars():
        #     name = var.varName
        #     if name.startswith('x['):
        #         (f,m,t) = map(int, name[2:-1].split(','))
        #         self.x[(f,m,t)] = var

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
        self._model.setObjective(self._objective_fun(c))

    def cal_obj(self, price, x):
        """
        Calculates the objective value for the scheduling problem.
        Supports price as (N,) or (B, N) and x as (Tasks, Machines, N) or (B, Tasks, Machines, N).
        """
        # 1. Standardize inputs to numpy for calculation
        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return np.asarray(data, dtype=np.float32)

        price = to_numpy(price)
        x = to_numpy(x)
        
        # Retrieve constants from your class instance
        P = np.array(self.P)    # Power consumption per task
        D = np.array(self.D)    # Duration per task
        q = self.q
        N = price.shape[-1]
        
        # 2. Pre-calculate the 'Cost of starting task f at time t'
        # For each task f, we need a sliding window sum of 'price' with window size D[f]
        # We can use np.convolve or a simple loop since D varies per task
        
        # price_kernels shape: (Tasks, N)
        # entry (f, t) is the sum of prices from t to t + D[f]
        if price.ndim == 1:
            price_kernels = np.array([
                np.convolve(price, np.ones(D[f]), mode='valid')[:N-D[f]+1] 
                for f in range(self.nbTasks)
            ], dtype=object) 
        else: # Batch mode (B, N)
            # Handle batching via list comprehension or vectorized operations
            price_kernels = []
            for b in range(price.shape[0]):
                b_kernels = [np.convolve(price[b], np.ones(D[f]), mode='valid')[:N-D[f]+1] 
                            for f in range(self.nbTasks)]
                price_kernels.append(b_kernels)
            price_kernels = np.array(price_kernels)

        # 3. Calculate Objective based on x shape
        if x.ndim == 1:
            # reshape x to (Tasks, Machines, N) if it's flat
            x = x.reshape(self.nbTasks, self.nbMachines, N)
        # Case 1: x is (Tasks, Machines, N)
        if x.ndim == 3:
            total_cost = 0
            for f in range(self.nbTasks):
                # Sum over machines for task f: (N,)
                x_f = np.sum(x[f, :, :N-D[f]+1], axis=0)
                # Cost = x_f * price_sum * Power * (q/60)
                total_cost += np.sum(x_f * price_kernels[f] * P[f])
            return total_cost * (q / 60)

        # Case 2: x is (B, Tasks, Machines, N)
        elif x.ndim == 4:
            batch_size = x.shape[0]
            batch_costs = np.zeros(batch_size)
            for b in range(batch_size):
                b_cost = 0
                for f in range(self.nbTasks):
                    x_f = np.sum(x[b, f, :, :N-D[f]+1], axis=0)
                    # Select the correct price kernel if price was (B, N) or (N,)
                    pk = price_kernels[b][f] if price.ndim == 2 else price_kernels[f]
                    b_cost += np.sum(x_f * pk * P[f])
                batch_costs[b] = b_cost * (q / 60)
            return batch_costs

        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")
        
    def _objective_fun(self, c):
        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        q= self.q
        N = 1440//q  

        nbMachines = self.nbMachines
        nbTasks = self.nbTasks
        nbResources = self.nbResources
        Machines = range(nbMachines)
        Tasks = range(nbTasks)
        Resources = range(nbResources)

        # c is the cost vector for each task, shape (Tasks,)
        # We need to create an expression that sums over all tasks, machines, and time slots
        
        obj_expr = gp.quicksum( [self.x[f,m,t]*sum(c[t:t+D[f]])*P[f]*q/60 
            for f in Tasks for t in range(N-D[f]+1) for m in Machines if (f,m,t) in self.x] )
        return obj_expr
    
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


def energy_generator_factory(instance = 1):
    def generator(num_data): #TODO: num_data is not yet implemented
        x_train, y_train, x_val, y_val, x_test, y_test = get_data()

        params = get_instance_config("data/load{}/day01.txt".format(instance))
        optmodel = SolveICON(**params)

        return x_train, y_train, x_val, y_val, x_test, y_test, optmodel
    return generator


if __name__ == "__main__":
    sizes = np.linspace(10, 350, 15).astype(int)
    sizes = np.linspace(200, 200, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=energy_generator_factory(),
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
