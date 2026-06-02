"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PyDFLT/PyDFLT
"""
import numpy as np
import torch
from pyepo.model.opt import optModel

from pyepo.dfl.predictor import Predictor
from pyepo.dfl.utils import EarlyStopper, set_seeds
from pyepo.dfl.DFLMaker import DFLMaker
import time
from torch.utils.data import TensorDataset

class MSEDecisionMaker(DFLMaker):
    def __init__(
        self,
        predictor: Predictor,
        optmodel: optModel,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 1000,
        seed: int | None = None,
        grouped: bool = False,
        device: str = "cpu",
    ) -> None:
        predictor.to(device)
        self.predictor = predictor
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.learning_rate = lr
        self.optmodel = optmodel
        self.early_stopper = EarlyStopper(patience=15, min_delta=0.01)
        self.grouped = grouped
        self.device = device

        self._set_optimizer()
        set_seeds(seed)

    def _set_optimizer(self) -> None:
        """
        Sets the optimizer based on the trainable parameter of the predictor and the learning rate
        """
        print(f"set learning rate to {self.learning_rate}")
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.learning_rate)

    def update(self, features: torch.Tensor, costs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Updates the predictive model using the MVD (Measure-Valued Derivative) gradient.

        Samples from the distributional predictor, evaluates the objective for each sample,
        computes the SFGE loss via the log-derivative trick, and performs one gradient step.

        Args:
            features (torch.Tensor): The input features.
            costs (torch.Tensor): The costs associated with each sample.

        Returns:
            dict[str, torch.Tensor]: Accumulated losses and diagnostics for the logger,
                containing keys 'loss', 'eval', 'solver_calls', and 'sigma'.
        """

        # Obtain the distributional predictor and sample
        pred = self.predictor.forward(features)

        loss_terms: torch.Tensor = torch.mean((pred - costs)**2)

        loss = loss_terms.mean(dim=0)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        log_dict = {
            "loss": loss.cpu().detach().numpy().astype(np.float32),
        }
        return log_dict
    
    def run_batch(self, features: torch.Tensor, costs: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Evaluates the predictor on a batch of data without updating the model.

        Args:
            features (torch.Tensor): The input features.
            costs (torch.Tensor): The costs associated with each sample.
            optimal_solutions (torch.Tensor): The optimal solutions.
            optimal_objectives (torch.Tensor): The optimal objectives.
            metrics (list[str] | None): List of additional metrics to evaluate. Defaults to None.
        """
        pred = self.predictor.forward(features)

        loss_terms = torch.mean((pred - costs)**2)
        loss = loss_terms.mean(dim=0)
        logger_loss = loss.detach().cpu().numpy().astype(np.float32)

        # Logging
        log_dict = {
            "loss": logger_loss,
        }
        return log_dict
        

    def run_epoch(self, mode: str, data_loader: torch.utils.data.DataLoader, epoch_num: int) -> list[dict[str, float]]:
        """
        Runs one complete epoch in the specified mode (train/validation/test).

        In train mode, updates the noisifier's cooling schedule if configured, then processes
        each batch by calling `update`. In validation/test mode, evaluates the predictor using
        `_get_batch_results`.

        Args:
            mode (str): The mode to run ('train', 'validation', or 'test').
            data_loader (torch.utils.data.DataLoader): The data loader for the current mode.
            epoch_num (int): The current epoch number, used to update the noisifier cooling scheme.
            metrics (list[str] | None): List of additional metrics to evaluate. Defaults to None.

        Returns:
            list[dict[str, float]]: List of result dictionaries, one per batch processed.
        """
        assert mode in [
            "train",
            "validation",
            "test",
        ], "Mode must be train/validation/test!"

        # Switch predictor and problem mode to train/evaluation
        self.predictor.train() if mode == "train" else self.predictor.eval()

        # Initialize dictionary with the results
        epoch_results = []

        # Run
        for batch in data_loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            if mode == "train":
                batch_results = self.update(x, y)
            else:
                batch_results = self.run_batch(x, y)
            mode_batch_results = {f"{mode}/{key}": val for key, val in batch_results.items()}
            mode_batch_results["batch_size"] = len(x)
            epoch_results.append(mode_batch_results)

        return epoch_results
    
    def train_model(self, x_train, y_train, x_val, y_val):
        if self.grouped: 
            x_train = x_train.reshape(-1, x_train.shape[-1])
            y_train = y_train.reshape(-1)
            x_val = x_val.reshape(-1, x_val.shape[-1])
            y_val = y_val.reshape(-1)
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()),
            batch_size=self.batch_size, shuffle=False
        )
        epoch_times = []
        for epoch in range(self.num_epochs):
            start_time = time.perf_counter()
            _ = self.run_epoch("train", train_loader, epoch)
            val_results = self.run_epoch("validation", val_loader, epoch)

            # Calculate weighted average validation loss
            total_val_loss = sum(res["validation/loss"] * res["batch_size"] for res in val_results)
            total_val_samples = sum(res["batch_size"] for res in val_results)
            avg_val_loss = total_val_loss / total_val_samples

            # Early stopping check
            if self.early_stopper.step(avg_val_loss, self.predictor):
                print(f"Early stopping at epoch {epoch} with validation loss {avg_val_loss:.4f}")
                break
            epoch_time = time.perf_counter() - start_time
            epoch_times.append(epoch_time)
        info = {"mean_epoch_time": np.mean(epoch_times),
                "final_epoch": epoch+1}
        
        self.predictor.eval()  # Set to evaluation mode after training
        return avg_val_loss, info
    
    def optimize(self, x, m = None): 
        # Predict
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        device = next(self.predictor.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            pred_costs = self.predictor.forward(x)

        
        self.optmodel.setObj(pred_costs)
        sol, obj = self.optmodel.solve()

        return sol, obj
