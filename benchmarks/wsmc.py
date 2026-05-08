from pyepo.data.gen_wsmc_data import gen_data_wsmc
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pyepo.model.grb import optGrbModel
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import KernelPrescription, LossType

import random
from itertools import chain, combinations

#TODO: check the recovery ratio thing, I do not currently use it

class WeightedSetMultiCover(optGrbModel):
    """
    This weighted set multi cover problem is designed such that all sets are relevant, i.e. there are no sets that
    are equal to another set but with higher costs, and no sets such that the union of some of its subsets is cheaper.
    It has an additional recourse action being able to recover costs from sets that are unused.
    """

    def __init__(
        self,
        num_items: int,
        num_covers: int,
        penalty: float,
        cover_costs_lb: int,
        cover_costs_ub: int,
        recovery_ratio: float = 0,
        seed: int = 5,
        silvestri2024: bool = True,
        density: float = 0.25,
        num_scenarios: int = 1,
    ):
        """
        Initializes the WeightedSetMultiCover model.

        Args:
            num_items (int): Number of items that need to be covered.
            num_covers (int): Number of available covers (sets).
            penalty (float): Penalty for unmet coverage requirements.
            cover_costs_lb (int): Lower bound for cover costs.
            cover_costs_ub (int): Upper bound for cover costs.
            recovery_ratio (float): Ratio for recovering costs from unused covers. Defaults to 0.
            seed (int): Random seed for reproducible generation. Defaults to 0.
            silvestri2024 (bool): Whether to use silvestri2024 parameter generation method. Defaults to False.
            density (float): Density of the item-cover matrix when using silvestri2024 method. Defaults to 0.25.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
        # Setting input parameters
        self.num_items = num_items
        self.num_covers = num_covers
        self.penalty = penalty
        self.cover_costs_lb = cover_costs_lb
        self.cover_costs_ub = cover_costs_ub
        self.recovery_ratio = recovery_ratio
        self.seed = seed
        self.silvestri2024 = silvestri2024
        self.density = density
        self.num_scenarios = num_scenarios

        # Setting additional model parameters
        if self.silvestri2024:
            self.cover_costs, self.item_cover_matrix = self._set_fixed_parameters_silvestri2024(cover_costs_lb, cover_costs_ub, density, seed)
        else:
            self.cover_costs, self.item_cover_matrix = self._set_fixed_parameters(cover_costs_lb, cover_costs_ub, seed)
        self.max_cover_costs = (self.item_cover_matrix * self.cover_costs).max(axis=1)

        super().__init__()

    @property
    def num_cost(self):
        """
        number of cost to be predicted
        """
        return 0

    def _getModel(self) -> tuple[gp.Model, dict[str, gp.MVar | gp.Var]]:
        """
        Creates the Gurobi optimization model for the weighted set multi-cover problem.
        This method defines the first and second stage variables, constraints, and objective function.

        Returns:
            tuple: A tuple containing the Gurobi model and the variables dictionary.
        """
        # Create a GP model
        gp_model = gp.Model("wsmc")
        vars_dict = {}
        second_stage_vars_dict = {}

        # Define variables
        # number of each cover that is picked
        x = gp_model.addMVar((self.num_covers,), vtype=GRB.INTEGER, name="select_cover")
        # Unmet coverage based on cover selection
        y = gp_model.addMVar((self.num_items, self.num_scenarios), vtype=GRB.INTEGER, name="unmet_coverage")
        second_stage_vars_dict["unmet_coverage"] = y

        # It is a minimization problem
        gp_model.modelSense = GRB.MINIMIZE

        self.second_stage_vars_dict = second_stage_vars_dict

        return gp_model, x
    
    def _update_second_stage_vars(self, required_scenarios: int):
        """
        Updates the unmet_coverage variables if the number of scenarios has changed.
        """
        if self.num_scenarios != required_scenarios:
            # Remove old variables if they exist
            if "unmet_coverage" in self.second_stage_vars_dict:
                self._model.remove(self.second_stage_vars_dict["unmet_coverage"])
            
            # Update the scenario count
            self.num_scenarios = required_scenarios
            
            # Create new variables
            new_y = self._model.addMVar(
                (self.num_items, self.num_scenarios), 
                vtype=GRB.INTEGER, 
                name="unmet_coverage"
            )
            self.second_stage_vars_dict["unmet_coverage"] = new_y
            
            # Update model to integrate new variables
            self._model.update()

    def setObj(self, cover_requirements: np.ndarray) -> None:
        # Check if we need to resize y for a single scenario
        self._update_second_stage_vars(1)
        
        x = self.x
        y = self.second_stage_vars_dict["unmet_coverage"]

        # Objective: /1 is redundant but keeps logic consistent
        obj = (self.cover_costs @ x) + gp.quicksum(
            self.penalty * self.max_cover_costs[i] * y[i, 0]
            for i in range(self.num_items)
        )
        self._model.setObjective(obj)

        # Remove existing constraints
        self._model.remove(self._model.getConstrs())


        self._model.addConstrs(
            gp.quicksum(self.item_cover_matrix[i, j] * x[j] for j in range(self.num_covers)) + y[i, k] >= cover_requirements[i]
            for i in range(self.num_items)
            for k in range(self.num_scenarios)
        )

    def setWeightObj(self, w, c):
        """
        Set a weighted objective for predictive prescriptions.

        Args:
            w (np.ndarray): shape (N,), weights for each sample
            c (np.ndarray): shape (N, C), cost vectors for each sample
        """
        self._update_second_stage_vars(len(w))
        # Obtain the coverage parameters
        x = self.x
        y = self.second_stage_vars_dict["unmet_coverage"]

        # Set objective (there are no first stage constraints in this problem)
        obj = gp.quicksum(self.cover_costs[j] * x[j] for j in range(self.num_covers)) + gp.quicksum(
            self.penalty * self.max_cover_costs[i] * y[i, k] * w[k] for i in range(self.num_items) for k in range(self.num_scenarios)
        )
        self._model.setObjective(obj)

        # Remove existing constraints
        self._model.remove(self._model.getConstrs())


        self._model.addConstrs(
            gp.quicksum(self.item_cover_matrix[i, j] * x[j] for j in range(self.num_covers)) + y[i, k] >= c[k, i]
            for i in range(self.num_items)
            for k in range(self.num_scenarios)
        )

    def cal_obj(self, c, x):
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        c = np.asarray(c, dtype=np.float32)

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)

        # 1. First stage cost: c^T * x
        # If x is (32, 159, 25) and cover_costs is (25,)
        # result is (32, 159)
        first_stage_costs = x @ self.cover_costs

        # 2. Calculate current coverage (Ax)
        # item_cover_matrix is (num_items, num_sets) -> (I, 25)
        # x is (..., 25)
        # We need (x @ A.T) to get (..., num_items)
        current_coverage = x @ self.item_cover_matrix.T
            
        # 3. Handle Broadcasting for c
        # If c is (32, 10) and current_coverage is (32, 159, 10)
        if c.ndim == 2 and current_coverage.ndim == 3:
            # Reshape c from (32, 10) to (32, 1, 10)
            c_expanded = c[:, np.newaxis, :]
        else:
            c_expanded = c

        # 4. Calculate Unmet Costs
        unmet_coverage = np.maximum(0, c_expanded - current_coverage)
        
        # Multiply by penalty vector (num_items,)
        # Result is (32, 159)
        penalty_vec = self.penalty * self.max_cover_costs
        unmet_costs = unmet_coverage @ penalty_vec
        
        # 4. Total Objective
        # Summing everything implies we want the total cost across all instances/scenarios
        total_obj = first_stage_costs + unmet_costs
        
        return total_obj

    def _set_fixed_parameters(self, cover_costs_lb: int, cover_costs_ub: int, seed: int):
        """
        Generates fixed parameters for the weighted set multi-cover problem.
        This method creates covers and their costs such that all sets are relevant.

        Args:
            cover_costs_lb (int): Lower bound for cover costs.
            cover_costs_ub (int): Upper bound for cover costs.
            seed (int): Random seed for reproducible generation.

        Returns:
            tuple: A tuple containing cover costs and item-cover matrix.
        """
        rng = np.random.default_rng(seed)
        random.seed(seed)

        # We iterate through all possible combinations
        cover_costs_dict = {}
        cover_costs = np.zeros(self.num_covers)
        item_cover_matrix = np.zeros((self.num_items, self.num_covers))
        cover_idx = 0
        num_items_to_cover = 0
        while cover_idx < self.num_covers:
            num_items_to_cover += 1
            # Generate all possible combinations of `num_ones` positions, go over them randomly
            combinations_of_positions = list(combinations(range(self.num_items), num_items_to_cover))
            if num_items_to_cover > 1:  # we shuffle the combinations for set that covers multiple items
                random.shuffle(combinations_of_positions)
            for ones_positions in combinations_of_positions:
                if cover_idx >= self.num_covers:
                    break  # Stop if we filled all columns
                item_cover_matrix[list(ones_positions), cover_idx] = 1
                if num_items_to_cover == 1:
                    costs = rng.integers(cover_costs_lb, cover_costs_ub + 1)
                else:  # num_items_to_cover > 1
                    subsets = list(chain.from_iterable(combinations(ones_positions, r) for r in range(1, len(ones_positions))))
                    disjoint_union_subsets = []
                    for r in range(1, len(subsets) + 1):
                        for combo in combinations(subsets, r):
                            # Check if they cover the ones_positions and are disjoint
                            if ones_positions == tuple(set().union(*combo)) and all(
                                set(x).isdisjoint(set(y)) for i, x in enumerate(combo) for y in combo[i + 1 :]
                            ):
                                disjoint_union_subsets.append(combo)
                    min_costs = max([cover_costs_dict[s] for s in subsets])
                    max_costs = min([sum([cover_costs_dict[s] for s in dus]) for dus in disjoint_union_subsets])
                    costs = int((max_costs + min_costs) / 2)
                    # np.random.randint(min_costs, max_costs)
                cover_costs[cover_idx] = costs
                cover_costs_dict[ones_positions] = costs
                cover_idx += 1

        return cover_costs, item_cover_matrix

    def _set_fixed_parameters_silvestri2024(self, cover_costs_lb: float, cover_costs_ub: float, density: float, seed: int):
        """
        Generates fixed parameters using the silvestri2024 method.
        This method creates covers and their costs using a different approach.

        Args:
            cover_costs_lb (float): Lower bound for cover costs.
            cover_costs_ub (float): Upper bound for cover costs.
            density (float): Target density for the item-cover matrix.
            seed (int): Random seed for reproducible generation.

        Returns:
            tuple: A tuple containing cover costs and item-cover matrix.
        """
        rng = np.random.default_rng(seed)
        cover_costs = rng.uniform(cover_costs_lb, cover_costs_ub, self.num_covers)
        item_cover_matrix = np.zeros((self.num_items, self.num_covers))

        for item in range(self.num_items):  # get two covers for each item
            cover_1 = rng.integers(0, self.num_covers)
            leftover_covers = [i for i in range(0, self.num_covers) if i != cover_1]
            cover_2 = leftover_covers[rng.integers(0, self.num_covers - 1)]
            item_cover_matrix[item, cover_1] = 1
            item_cover_matrix[item, cover_2] = 1
        for cover in range(self.num_covers):  # cover an item
            item = rng.integers(0, self.num_items)
            item_cover_matrix[item, cover] = 1

        # add until density is reached, note that with small problem cases density is often already a lot higher
        while item_cover_matrix.mean() < density:
            item = rng.integers(0, self.num_items)
            cover = rng.integers(0, self.num_covers)
            item_cover_matrix[item, cover] = 1

        return cover_costs, item_cover_matrix
    


def wsmc_generator_factory(num_feat=5, num_item=10, num_sets=25):
    def generator(num_data, seed=42):
        x, c = gen_data_wsmc(seed=seed, num_data=num_data, num_features=num_feat, num_items=num_item, degree=5, noise_width=0.5)

        x_tmp, x_test, c_tmp, c_test = train_test_split(
            x, c, test_size=0.1, random_state=0 
        )

        x_train, x_val, c_train, c_val = train_test_split(
            x_tmp, c_tmp, test_size=0.11, random_state=0 
        )

        optmodel = WeightedSetMultiCover(
            num_items=num_item,
            num_covers=num_sets,
            penalty=5,
            cover_costs_lb=1,
            cover_costs_ub=10,
            recovery_ratio=0, #TODO: in paper this was 0.8
            seed=42,
            silvestri2024=True,
            density=0.25,
            num_scenarios=len(x_train)
        )
        return x_train, c_train, x_val, c_val, x_test, c_test, optmodel, {}
    return generator

if __name__ == "__main__":
    sizes = np.linspace(20, 20, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=wsmc_generator_factory(),
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
        "dropout": [0, 0.1],
        "num_hidden_layers": [0,1,2],
    }

    train_param_grid = {
        "epochs": [1000],
        "batch_size": [32],
        "lr": [1e-3, 5e-4],
    }

    dfl_model_param_grid = {
        "hidden_dim": [32, 64, 128],
        "dropout": [0, 0.1],
        "num_hidden_layers": [0,1,2],
    }

    pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{Rec.-KR}_N(x)$', WeightingTypeFunction.RKERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER, weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid) # Discrete Expectation Regret
    pipeline.add_model(r'$z^{SFGE}(x)$',  WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)


    pipeline.execute(save_dir="saved_models/wsmc/")
    # pipeline.plot_results('results/wsmc/wsmc_regret.png', 'Knapsack Benchmark Regret')
    pipeline.plot_boxplot(sizes[0], 'results/wsmc/wsmc_boxplot.png', 'Knapsack Benchmark Boxplot')
    # pipeline.plot_weight_distribution(150, 'results/wsmc/wsmc_weights.png', 'Knapsack Weight distribution')
