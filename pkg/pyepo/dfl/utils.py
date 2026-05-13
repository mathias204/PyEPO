
import copy
import numpy as np
import random
import torch


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state_dict = None

    def step(self, validation_loss, model):
        if validation_loss < self.min_validation_loss * (1 - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # restore best weights and stop
                if self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                return True  # stop training
            return False   
        
def set_seeds(seed: int, full_reproducibility_GPUs: bool = False) -> None:
    """
    Sets random seed for different packages.

    Args:
        seed (int): Random seed.
        full_reproducibility_GPUs (bool): Set to True to enable full GPU reproducibility. Might impact performance.
    """
    if seed is None:
        return

    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs if multiple are present
        if full_reproducibility_GPUs:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
