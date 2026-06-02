"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PyDFLT/PyDFLT
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pyepo.model.grb import optGrbModel
from sklearn.model_selection import train_test_split
import torch
from pyepo.eval.optimize_pipeline import PredictOptimizePipeline
from pyepo.predictive.utils import WeightingTypeFunction
from pyepo.predictive import LossType
from pyepo.hyperparameters import k_param_grid, kernel_param_grid, rf_param_grid, weight_model_param_grid, train_param_grid, dfl_model_param_grid

from itertools import combinations

from pyepo.data.ptsp import gen_data_ptsp


class TwoStagePTSP(optGrbModel):
    def __init__(
        self,
        num_cities: int,
        missed_city_penalty: float,
        recovery_ratio: float,
        radius: float = 10,
        noise_std: float = 1,
        seed: int = 5,
        num_scenarios: int = 1,
    ):
        self.num_cities = num_cities
        self.missed_city_penalty = missed_city_penalty
        self.recovery_ratio = recovery_ratio
        self.radius = radius
        self.noise_std = noise_std
        self.seed = seed
        self.num_scenarios = num_scenarios

        self.rng = np.random.default_rng(self.seed)
        self.num_nodes = self.num_cities + 1
        self.x_coord, self.y_coord = self._get_coords()
        self.distances = self._determine_distances()

        super().__init__()        

        self.lazy_constraints_method = self.subtourelim
        self._model.Params.lazyConstraints = 1

    @property
    def num_cost(self):
        """
        number of cost to be predicted
        """
        return None


    def _getModel(self):
        model = gp.Model()

        vars_dict = {}
        second_stage_vars_dict = {}
        auxiliary_vars_dict = {}


        arc_traversed = model.addMVar((self.num_nodes, self.num_nodes), name="x_arc", vtype=GRB.BINARY)
        direct_trip = model.addMVar((self.num_cities,), name="x_direct", vtype=GRB.BINARY)
        city_visited = model.addMVar((self.num_cities,), name="x_visited", vtype=GRB.BINARY)
        tour_exists = model.addMVar((1,), name="x_tour", vtype=GRB.BINARY)
        city_canceled = model.addMVar((self.num_cities, self.num_scenarios), name="y_canceled", vtype=GRB.BINARY)

        model.modelSense = GRB.MINIMIZE

        # Enforce arc symmetry and no self-loops
        model.addConstrs(arc_traversed[i, j] == arc_traversed[j, i] for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes))
        model.addConstrs(arc_traversed[i, i] == 0 for i in range(self.num_nodes))

        # A direct trip implies the city is visited
        model.addConstrs(city_visited[i] >= direct_trip[i] for i in range(self.num_cities))

        # Degree-2 constraint for cities visited via the tour (not direct trips)
        model.addConstrs(
            gp.quicksum(arc_traversed[i + 1, j] for j in range(self.num_nodes)) == 2 * (city_visited[i] - direct_trip[i]) for i in range(self.num_cities)
        )
        # Depot has degree 2 iff a tour exists
        model.addConstr(gp.quicksum(arc_traversed[0, j] for j in range(self.num_nodes)) == 2 * tour_exists[0])

        # Tour exists iff at least one city is visited via the tour
        model.addConstr(tour_exists[0] >= gp.quicksum(city_visited[i] - direct_trip[i] for i in range(self.num_cities)) / self.num_cities)

        vars_dict["x_arc"] = arc_traversed
        vars_dict["x_direct"] = direct_trip
        auxiliary_vars_dict["x_visited"] = city_visited
        auxiliary_vars_dict["x_tour"] = tour_exists
        second_stage_vars_dict["y_canceled"] = city_canceled

        model.modelSense = GRB.MINIMIZE

        self.second_stage_vars_dict = second_stage_vars_dict
        self.auxiliary_vars_dict = auxiliary_vars_dict

        return model, vars_dict

    def _update_second_stage_vars(self, required_scenarios: int):
        # Remove existing cancellation constraints
        if len(self._model.getConstrs()):
            constraints_to_remove = []
            for k in range(self.num_scenarios):
                for i in range(self.num_cities):
                    c = self._model.getConstrByName(f"canceled[{i},{k}]")
                    if c is not None:
                        constraints_to_remove.append(c)
            for c in constraints_to_remove:
                self._model.remove(c)

        if self.num_scenarios != required_scenarios:
            # Remove existing second-stage variables and constraints
            for k in range(self.num_scenarios):
                for i in range(self.num_cities):
                    var = self._model.getVarByName(f"y_canceled[{i},{k}]")
                    if var is not None:
                        self._model.remove(var)
            self._model.update()

            # Add new second-stage variables
            city_canceled = self._model.addMVar((self.num_cities, required_scenarios), name="y_canceled", vtype=GRB.BINARY)
            self.second_stage_vars_dict["y_canceled"] = city_canceled

            self.num_scenarios = required_scenarios


    def cal_obj(self, c, x):
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        c = np.asarray(c, dtype=np.float32)

        # Handle nested lists/arrays of dictionaries
        if isinstance(x, (list, np.ndarray)):

            x_arr = np.asarray(x, dtype=object)
            original_shape = x_arr.shape

            flat_x = x_arr.reshape(-1)

            arc_traversed = np.stack(
                [
                    np.asarray(sol["x_arc"], dtype=np.float32)
                    for sol in flat_x
                ],
                axis=0
            )

            direct_trip = np.stack(
                [
                    np.asarray(sol["x_direct"], dtype=np.float32)
                    for sol in flat_x
                ],
                axis=0
            )

            # Restore batch dimensions
            arc_traversed = arc_traversed.reshape(
                *original_shape,
                *arc_traversed.shape[1:]
            )

            direct_trip = direct_trip.reshape(
                *original_shape,
                *direct_trip.shape[1:]
            )

        else:
            arc_traversed = np.asarray(x["x_arc"], dtype=np.float32)
            direct_trip = np.asarray(x["x_direct"], dtype=np.float32)

        # Determine visitation requirements based on predictions
        requires_visit = np.round(np.clip(c, 0, 1))

        # Handle broadcasting for additional solution dimensions
        while requires_visit.ndim < direct_trip.ndim:
            requires_visit = np.expand_dims(requires_visit, axis=-2)

        requires_visit_exp = requires_visit

        # Derive city_visited from routing variables
        # Degree 2 means the city is part of the tour
        node_degrees = (
            np.sum(arc_traversed, axis=-1)[..., 1:]
            + np.sum(arc_traversed, axis=-2)[..., 1:]
        )

        visited_by_tour = np.clip(node_degrees / 2, 0, 1)

        # A city is visited either by the tour or by direct trip
        city_visited = np.clip(
            visited_by_tour + direct_trip,
            0,
            1
        )

        # 1. Base arc costs
        upper_distances = np.triu(self.distances, k=1)

        if (
            arc_traversed.shape[-1] == self.num_nodes
            and arc_traversed.ndim >= 2
        ):
            arc_cost = np.sum(
                arc_traversed * upper_distances,
                axis=(-2, -1)
            )
        else:
            # Fallback for flattened upper-triangular edge arrays
            distances_flat = upper_distances[
                np.triu_indices(self.num_nodes, k=1)
            ]

            arc_cost = arc_traversed @ distances_flat

        # 2. Direct trip costs
        dt_distances = 2 * self.distances[0, 1:]

        direct_trip_cost = direct_trip @ dt_distances

        # 3. Second-stage penalty costs
        penalty_vec = (
            self.missed_city_penalty
            * 2
            * self.distances[0, 1:]
        )

        missed_cities = (
            requires_visit_exp
            * (1 - city_visited)
        )

        penalty_cost = missed_cities @ penalty_vec

        # 4. Second-stage cancellation savings
        y_canceled = (
            direct_trip
            * (1 - requires_visit_exp)
        )

        savings_vec = (
            self.recovery_ratio
            * 2
            * self.distances[0, 1:]
        )

        cancellation_savings = y_canceled @ savings_vec

        # Total Objective Function
        total_obj = (
            arc_cost
            + direct_trip_cost
            + penalty_cost
            - cancellation_savings
        )

        return total_obj
    
    def setObj(self, c):
        self._update_second_stage_vars(1)

        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()

        requires_visit = np.round(np.clip(c, 0, 1))  # shape: (num_cities,)

        arc_traversed = self.x["x_arc"]
        direct_trip = self.x["x_direct"]
        city_visited = self.auxiliary_vars_dict["x_visited"]
        city_canceled = self.second_stage_vars_dict["y_canceled"]

        assert city_canceled.shape[1] == 1, "Expected only one scenario for setObj"

        # Vectorized cancellation constraints
        allow_cancel = (1 - requires_visit)[:, None]  # (num_cities, 1)
        self._model.addConstr(
            city_canceled <= direct_trip[:, None] * allow_cancel,
            name="canceled"
        )

        # --- Precompute numpy coefficient arrays ---
        direct_dist = 2 * self.distances[0, 1:]                 # shape: (num_cities,)
        upper = np.triu(self.distances, k=1)                     # shape: (num_nodes, num_nodes)
        missed_coeff = requires_visit * self.missed_city_penalty * direct_dist  # shape: (num_cities,)
        recovery_coeff = (self.recovery_ratio * direct_dist)[:, None]           # shape: (num_cities, 1)

        # --- Build objective ---
        obj = (
            (upper * arc_traversed).sum()           # arc traversal costs
            + direct_dist @ direct_trip             # direct trip costs
            + missed_coeff @ (1 - city_visited)     # missed city penalties
            - (recovery_coeff * city_canceled).sum() # recovery savings
        )

        self._model.setObjective(obj)

    def setWeightObj(self, w, c):
        self._update_second_stage_vars(w.shape[0])

        requires_visit = np.round(np.clip(c, 0, 1))

        arc_traversed = self.x["x_arc"]
        direct_trip = self.x["x_direct"]
        city_visited = self.auxiliary_vars_dict["x_visited"]
        city_canceled = self.second_stage_vars_dict["y_canceled"]

        allow_cancel = (1 - requires_visit.T)  # (num_cities, num_scenarios)
        self._model.addConstr(
            city_canceled <= direct_trip[:, None] * allow_cancel,
            name="canceled"
        )

        direct_dist = 2 * self.distances[0, 1:]  # shape: (num_cities,)

        upper = np.triu(self.distances, k=1)  # shape: (num_nodes, num_nodes), lower zeroed

        missed_coeff = (w @ requires_visit) * self.missed_city_penalty * direct_dist
        recovery_coeff = self.recovery_ratio * direct_dist[:, None] * w[None, :]

        # --- Build objective using matrix operations ---
        obj = (
            (upper * arc_traversed).sum()                        # arc traversal costs
            + direct_dist @ direct_trip                          # direct trip costs
            + missed_coeff @ (1 - city_visited)                  # missed city penalties
            - (recovery_coeff * city_canceled).sum()             # recovery savings
        )

        self._model.setObjective(obj)

    def _get_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates node coordinates: cities on a noisy circle, depot at the origin.

        Returns:
            tuple: ``(x_coord, y_coord)`` arrays of length ``num_nodes``, with the
                depot at index 0.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_cities, endpoint=False)
        noise = self.rng.normal(0, self.noise_std, self.num_cities)
        perturbed_x = (self.radius + noise) * np.cos(angles)
        perturbed_y = (self.radius + noise) * np.sin(angles)
        x_coord = np.insert(perturbed_x, 0, 0.0)
        y_coord = np.insert(perturbed_y, 0, 0.0)
        return x_coord, y_coord

    def _determine_distances(self) -> np.ndarray:
        """
        Computes the pairwise Euclidean distance matrix for all nodes.

        Returns:
            np.ndarray: Distance matrix of shape ``(num_nodes, num_nodes)``.
        """
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                distances[i, j] = np.sqrt((self.x_coord[i] - self.x_coord[j]) ** 2 + (self.y_coord[i] - self.y_coord[j]) ** 2)
        return distances
    
    @staticmethod
    def subtourelim(model, where):
        """
        Gurobi lazy-constraint callback for subtour elimination.

        Finds the shortest cycle in the current MIP solution and, if it is shorter
        than the full tour, adds a subtour-elimination constraint via ``cbLazy``.

        Args:
            model: The Gurobi model passed by the callback mechanism.
            where: The Gurobi callback location code.
        """
        if where == GRB.Callback.MIPSOL:
            arc_vars = [var for var in model.getVars() if "x_arc" in var.VarName]
            vals = model.cbGetSolution(arc_vars)
            selected = gp.tuplelist((var.VarName[-4], var.VarName[-2]) for i, var in enumerate(arc_vars) if vals[i] > 0.5)

            direct_vars = [var for var in model.getVars() if "x_direct" in var.VarName]
            direct_vals = model.cbGetSolution(direct_vars)
            direct_trips = sum(direct_vals[i] for i in range(len(direct_vals)))
            visited_vars = [var for var in model.getVars() if "x_visited" in var.VarName]
            visited_vals = model.cbGetSolution(visited_vars)
            visited_cities = sum(visited_vals[i] for i in range(len(visited_vals)))
            to_visit_with_tour = visited_cities - direct_trips

            unvisited = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]
            tour = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]
            unvisited.insert(0, 0)
            tour.insert(0, 0)
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in selected.select(current, "*") if j in unvisited]
                if len(thiscycle) <= len(tour):
                    tour = thiscycle

            if len(tour) < to_visit_with_tour:
                model.cbLazy(gp.quicksum(arc_vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)


def ptsp_generator_factory(feats=5, customers=10, degree=5):
    def generator(num_data, seed=42):

        x, c = gen_data_ptsp(seed, num_data=num_data, num_features=feats, num_customers=customers, degree=degree, noise_width=0.5, scale=0.3)
        x_train, x_tmp, c_train, c_tmp = train_test_split(
            x, c, test_size=0.2, random_state=seed 
        )
        x_val, x_test, c_val, c_test = train_test_split(
            x_tmp, c_tmp, test_size=0.5, random_state=seed 
        )

        model = TwoStagePTSP(num_cities=customers, missed_city_penalty=5, recovery_ratio=1, radius=10, noise_std=5, seed=seed, num_scenarios=len(x_train))

        return x_train, c_train, x_val, c_val, x_test, c_test, model, {}
    return generator



if __name__ == "__main__":
    gp.setParam("OutputFlag", 0)
    
    sizes = np.linspace(500, 500, 1).astype(int)
    
    pipeline = PredictOptimizePipeline(
        data_sizes=sizes, 
        data_generator=ptsp_generator_factory(),
        num_runs=5
    )


    train_param_grid = {
        **train_param_grid,
        "epochs": [5000],
        "batch_size": [32],
    }

    pipeline.add_model(r'$\hat{z}^{kNN}_N(x)$', WeightingTypeFunction.NEAREST_NEIGHBOUR, param_grid = k_param_grid)
    pipeline.add_model(r'$\hat{z}^{KR}_N(x)$', WeightingTypeFunction.KERNEL, param_grid = kernel_param_grid)
    pipeline.add_model(r'$\hat{z}^{RF}_N(x)$', WeightingTypeFunction.RANDOM_FOREST, param_grid = rf_param_grid)
    pipeline.add_model(r'$\hat{z}^{DER}_N(x)$',  WeightingTypeFunction.NEURAL, loss=LossType.DER,      weight_model_param_grid=weight_model_param_grid, train_param_grid=train_param_grid) # Discrete Expectation Regret

    pipeline.add_model(r'$z^{SFGE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.SFGE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)
    pipeline.add_model(r'$z^{MSE}(x)$', WeightingTypeFunction.NEURAL_DFL, loss=LossType.MSE, dfl_predictor_param_grid=dfl_model_param_grid, train_param_grid=train_param_grid)

    # # Run and plot
    pipeline.execute(save_dir="saved_models/ptsp/", force_run=True)
    pipeline.save_results_to_csv('results/ptsp/optimize_results.csv')
    pipeline.plot_boxplot(sizes[0], 'results/ptsp/ptsp_boxplot_test.png', 'Weighted Set Multi-Cover Benchmark Boxplot')
