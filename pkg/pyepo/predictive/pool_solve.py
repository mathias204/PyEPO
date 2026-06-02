import torch
from pyepo.utlis import getArgs


def solve_in_pass(cp, optmodel, processes, pool):
    """
    A function to solve optimization in the forward/backward pass
    """
    # get device
    device = cp.device
    # number of instance
    ins_num = len(cp)

    # single-core
    if processes == 1:
        sol = []
        obj = []
        for i in range(ins_num):
            # solve
            optmodel.setObj(cp[i])
            solp, objp = optmodel.solve()
            sol.append(torch.as_tensor(solp))
            obj.append(objp)
        # to tensor
        sol = torch.stack(sol, dim=0).to(device)
        obj = torch.tensor(obj, dtype=torch.float32, device=device)
    # multi-core
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        res = pool.amap(_solveWithObj4Par, cp, [args] * ins_num,
                        [model_type] * ins_num).get()
        # get res
        if isinstance(res[0][0], torch.Tensor):
            sol = torch.stack([r[0] for r in res], dim=0).to(device)
        else:
            sol = [r[0] for r in res]
        obj = torch.tensor([r[1] for r in res], dtype=torch.float32, device=device)
    return sol, obj


def _solveWithObj4Par(cost, args, model_type):
    """
    A function to solve function in parallel processors

    Args:
        cost (np.ndarray): cost of objective function
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        tuple: optimal solution (list) and objective value (float)
    """
    # rebuild model
    optmodel = model_type(**args)
    # set obj
    optmodel.setObj(cost)
    # solve
    sol, obj = optmodel.solve()
    # to tensor
    if isinstance(sol, list):
        sol = torch.tensor(sol, dtype=torch.float32)
    return sol, obj