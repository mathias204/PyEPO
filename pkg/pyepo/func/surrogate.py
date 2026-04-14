#!/usr/bin/env python
# coding: utf-8
"""
Surrogate Loss function
"""
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.model.opt import optModel
from pyepo.func.utlis import _solve_or_cache

class SPOPlus(optModule):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO
    (regret) Loss, which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build carterion
        self.spop = SPOPlusFunc()

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss


class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        c = true_cost.detach()
        w = true_sol.detach()
        z = true_obj.detach()
        # check sol
        #_check_sol(c, w, z)
        # solve
        sol, obj = _solve_or_cache(2 * cp - c, module)
        # calculate loss
        if module.optmodel.modelSense == EPO.MINIMIZE:
            loss = - obj + 2 * torch.einsum("bi,bi->b", cp, w) - z.squeeze(dim=-1)
        elif module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = obj - 2 * torch.einsum("bi,bi->b", cp, w) + z.squeeze(dim=-1)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optmodel = ctx.optmodel
        if optmodel.modelSense == EPO.MINIMIZE:
            grad = 2 * (w - wq)
        elif optmodel.modelSense == EPO.MAXIMIZE:
            grad = 2 * (wq - w)
        else:
            raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        return grad_output.unsqueeze(1) * grad, None, None, None, None


class perturbationGradient(optModule):
    """
    An autograd module for PG Loss, as a surrogate loss function of objective
    value, which measures the decision quality of the optimization problem.

    For PG Loss, the objective function is linear, and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    According to Danskin’s Theorem, the PG Loss is derived from different zeroth
    order approximations and has the informative gradient. Thus, it allows us to
    design an algorithm based on stochastic gradient descent.

    Reference: <https://arxiv.org/abs/2402.03256>
    """
    def __init__(self, optmodel, sigma=0.1, two_sides=False, processes=1, solve_ratio=1,
                 reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            sigma (float): the amplitude of the finite difference width used for loss approximation
            two_sides (bool): approximate gradient by two-sided perturbation or not
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # finite difference width
        self.sigma = sigma
        # symmetric perturbation
        self.two_sides = two_sides

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        loss = self._finiteDifference(pred_cost, true_cost)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss

    def _finiteDifference(self, pred_cost, true_cost):
        """
        Zeroth order approximations for surrogate objective value
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach()
        c = true_cost.detach()
        # central differencing
        if self.two_sides:
            # solve
            wp, _ = _solve_or_cache(cp + self.sigma * c, self)
            wm, _ = _solve_or_cache(cp - self.sigma * c, self)
            # convert numpy
            sol_plus = torch.as_tensor(wp, dtype=torch.float, device=device)
            sol_minus = torch.as_tensor(wm, dtype=torch.float, device=device)
            # differentiable objective value
            obj_plus = torch.einsum("bi,bi->b", pred_cost + self.sigma * true_cost, sol_plus)
            obj_minus = torch.einsum("bi,bi->b", pred_cost - self.sigma * true_cost, sol_minus)
            # loss
            if self.optmodel.modelSense == EPO.MINIMIZE:
                loss = (obj_plus - obj_minus) / (2 * self.sigma)
            elif self.optmodel.modelSense == EPO.MAXIMIZE:
                loss = (obj_minus - obj_plus) / (2 * self.sigma)
            else:
                raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        # back differencing
        else:
            # solve
            w, _ = _solve_or_cache(cp, self)
            wm, _ = _solve_or_cache(cp - self.sigma * c, self)
            # convert numpy
            sol = torch.as_tensor(w, dtype=torch.float, device=device)
            sol_minus = torch.as_tensor(wm, dtype=torch.float, device=device)
            # differentiable objective value
            obj = torch.einsum("bi,bi->b", pred_cost, sol)
            obj_minus = torch.einsum("bi,bi->b", pred_cost - self.sigma * true_cost, sol_minus)
            # loss
            if self.optmodel.modelSense == EPO.MINIMIZE:
                loss = (obj - obj_minus) / self.sigma
            elif self.optmodel.modelSense == EPO.MAXIMIZE:
                loss = (obj_minus - obj) / self.sigma
            else:
                raise ValueError("Invalid modelSense. Must be EPO.MINIMIZE or EPO.MAXIMIZE.")
        return loss
    

def SFGE(weights: torch.Tensor, true_cost: torch.Tensor, true_obj: torch.Tensor, data_sols:torch.Tensor, model: optModel, S) -> torch.Tensor:
    S 
    B, N = weights.shape
    device = weights.device
    cat = torch.distributions.Categorical(probs=weights)

    indices = cat.sample((S,)).t()  # (B, S)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, S)  # (B, S)
    sampled_sols = data_sols[batch_idx, indices] # (B, S, num_vars)
    
    realised_obj = model.cal_obj(true_cost, sampled_sols)
    realised_obj = torch.tensor(realised_obj, dtype=torch.float32).to(device) # (B, S)
    true_obj = true_obj.expand(-1, S) # (B, S)

    assert true_obj.shape == realised_obj.shape    

    if model.modelSense == EPO.MINIMIZE:
        loss: torch.Tensor =  (realised_obj - true_obj) * (cat.log_prob(indices.t())).t()
    else:
        # print("MAXIMIZE")
        loss: torch.Tensor = (true_obj - realised_obj) * (cat.log_prob(indices.t())).t()

    return loss.mean()

def DER(weights: torch.Tensor, true_cost: torch.Tensor, true_obj: torch.Tensor, data_sols:torch.Tensor, model: optModel) -> torch.Tensor:
    realised_obj = model.cal_obj(true_cost, data_sols)
    realised_obj = torch.tensor(realised_obj, dtype=torch.float32).to(weights.device)

    if model.modelSense == EPO.MINIMIZE:
        loss: torch.Tensor =  (realised_obj - true_obj) * weights
    else:
        loss: torch.Tensor = (true_obj - realised_obj) * weights  

    loss /= (true_obj + 1e-8)

    final_loss = loss.sum()
    return final_loss

# def novel(weights: torch.Tensor, true_cost: torch.Tensor, true_obj: torch.Tensor, data_sols: torch.Tensor, model: optModel) -> torch.Tensor:
#     # 1. Calculate raw costs (No Grad)
#     with torch.no_grad():
#         realised_obj = model.cal_obj(true_cost, data_sols)
#         # realised_obj shape: [300] (or however many solutions you have)
#         costs = torch.tensor(realised_obj, dtype=torch.float32).to(weights.device)

#         # 2. CRITICAL: Normalize costs. 
#         # This acts as a baseline. Positive values = "bad", Negative = "good".
#         # If we don't do this, gradients are dominated by the raw magnitude of the cost.
#         cost_std, cost_mean = torch.std_mean(costs)
#         # Avoid division by zero
#         normalized_costs = (costs - cost_mean) / (cost_std + 1e-8)
        
#         # If minimizing, we want to push probability UP for low costs (negative normalized values)
#         # and DOWN for high costs (positive normalized values).
#         if model.modelSense != EPO.MINIMIZE:
#             normalized_costs = -normalized_costs

#     # 3. Calculate Expected Risk
#     # We detach costs because we only differentiate with respect to weights (p)
#     loss = (weights * normalized_costs).sum()
    
#     return loss
