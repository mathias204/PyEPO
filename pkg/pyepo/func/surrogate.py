#!/usr/bin/env python
# coding: utf-8
"""
Surrogate Loss function
"""
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
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
    


class StochasticSmoothingLoss(optModule):
    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="sum", dataset=None, S = 800):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build criterion
        self.S = S
        self.ssl = StochasticSmoothingLossFun()


    def forward(self, weights, true_cost, true_obj, data_sols):
        loss = self.ssl.apply(weights, true_cost, true_obj, data_sols, self)
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
    
class StochasticSmoothingLossFun(Function):
    @staticmethod
    def forward(ctx, weights, true_cost, true_obj, data_sols, module: optModule):
        ctx.weights = weights
        ctx.true_cost= true_cost
        ctx.true_obj = true_obj
        ctx.module = module   
        ctx.data_sols = data_sols   
        device = weights.device    #TODO: save the tensors using ctx.save_for_backward

        B, N = weights.shape  # N = number of candidates
        S = module.S

        indices = torch.multinomial(weights, S, replacement=True).to(device)  # (B, S)
        print(indices.requires_grad)
        ctx.save_for_backward(indices)
        batch_idx = torch.arange(B, device=weights.device).unsqueeze(1).expand(B, S)  # (B, S)
        sampled_sols = data_sols[batch_idx, indices]  # (B, S, num_vars)

        realised_obj = module.optmodel.cal_obj(true_cost, sampled_sols)
        realised_obj = torch.tensor(realised_obj, dtype=torch.float32).to(true_cost.device)

        realised_obj = realised_obj.mean(dim=1)

        true_obj = true_obj.squeeze()

        assert realised_obj.shape == true_obj.shape
        if module.optmodel.modelSense == EPO.MINIMIZE:
            loss =  realised_obj - true_obj
        else:
            loss = true_obj - realised_obj
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        #TODO: not finished
        weights = ctx.weights
        B, N = ctx.weights.shape  # N = number of candidates
        S = ctx.module.S
        device = weights.device
        indices, = ctx.saved_tensors
        assert indices.min() >= 0 and indices.max() < N
        batch_idx = torch.arange(B, device=weights.device).unsqueeze(1).expand(B, S)  # (B, S)

        sampled_sols = ctx.data_sols[batch_idx, indices]  # (B, S, num_vars)
    
        realised_objs = ctx.module.optmodel.cal_obj(ctx.true_cost, sampled_sols)
        realised_objs = torch.tensor(realised_objs, dtype=torch.float32).to(device)
        assert realised_objs.shape == (B, S)

        true_obj = ctx.true_obj  # (B, 1)
        true_obj = true_obj.expand(-1, S) # (B, S)
    
        assert realised_objs.shape == true_obj.shape
        if ctx.module.optmodel.modelSense == EPO.MINIMIZE:
            L_i : torch.Tensor =  realised_objs - true_obj 
        else:
            L_i = true_obj - realised_objs 
        assert L_i.shape == (B, S)

        baseline = L_i.mean(dim=1, keepdim=True) # This is the average over the samples S, TODO: verify if this is correct
        baseline = baseline.expand(-1, S)
        assert baseline.shape == (B, S)
        L_i = (L_i - baseline)

        gradients = torch.zeros(ctx.weights.shape).to(device)


        w_sel = weights[batch_idx, indices]          # shape (B,)
        log_w = torch.log(w_sel)              # scalar  
        test = torch.autograd.grad(log_w, weights, grad_outputs=torch.ones_like(log_w), retain_graph=True)[0]
        gradients += torch.sum(L_i * test, dim=1)

        gradients /= ctx.module.S

        return gradients, None, None, None, None