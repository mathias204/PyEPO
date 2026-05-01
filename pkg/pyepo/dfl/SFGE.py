import numpy as np
import torch
from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo import EPO

from pyepo.dfl.noisifier import Noisifier
from pyepo.dfl.utils import EarlyStopper
from pyepo.dfl.DFLMaker import DFLMaker

class SFGEDecisionMaker(DFLMaker):
    """
    The SFGE decision maker is based on the paper "Silvestri, M., Berden, S., Mandi, J., Mahmutogullari, A. I., Amos, B.,
    Guns, T., & Lombardi, M. (2023). Score Function Gradient Estimation to Widen the Applicability of Decision-Focused
    Learning. arXiv preprint arXiv:2307.05213". The SFGE approach overcomes the zero-gradient problem in DFL by using
    a stochastic predictor at training time to smoothen the regret loss. In this codebase, we use the object Noisifier
    as the smoothing distribution around the parameterized predictive model that can be any from the list "allowed_predictors".
    Note that the Noisifier parameters are passed through "noisifier_kwargs",  which include the sigma setting.
    """

    def __init__(
        self,
        noisifier: Noisifier,
        optmodel: optModel,
        batch_size: int = 32,
        lr: float = 1e-3,
        standardize_loss: bool = True,
        epochs: int = 1000,
        num_samples: int = 1,           # Variable S
    ) -> None:
        self.num_samples = num_samples
        self.standardize_loss = standardize_loss
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.learning_rate = lr
        self.noisifier = noisifier
        self.optmodel = optmodel
        self.early_stopper = EarlyStopper(patience=5, min_delta=0)
        self._set_optimizer()

    def _set_optimizer(self) -> None:
        """
        Sets the optimizer based on the trainable parameter of the predictor and the learning rate
        """
        print(f"set learning rate to {self.learning_rate}")
        self.optimizer = torch.optim.Adam(self.noisifier.parameters(), lr=self.learning_rate)

    def update(self, features: torch.Tensor, costs: torch.Tensor, optimal_objectives: torch.Tensor, epsilon: float = 10**-5) -> dict[str, torch.Tensor]:
        """
        Updates the predictive model using the MVD (Measure-Valued Derivative) gradient.

        Samples from the distributional predictor, evaluates the objective for each sample,
        computes the SFGE loss via the log-derivative trick, and performs one gradient step.

        Args:
            features (torch.Tensor): The input features.
            costs (torch.Tensor): The costs associated with each sample.
            optimal_objectives (torch.Tensor): The optimal objective values.
            epsilon (float): Unused parameter kept for API compatibility. Defaults to 1e-5.

        Returns:
            dict[str, torch.Tensor]: Accumulated losses and diagnostics for the logger,
                containing keys 'loss', 'eval', 'solver_calls', and 'sigma'.
        """

        # Obtain the distributional predictor and sample
        distribution = self.noisifier.forward_dist(features)
        samples = distribution.sample((self.num_samples,))  # Shape: (num_samples, batch_size, num_parameters)

        # Get log probabilities
        individual_log_probs = distribution.log_prob(samples)
        log_probs = individual_log_probs.sum(dim=-1)  # sum over num_parameters dimension (the last one)

        # Get objective value per sample
        objectives = torch.zeros(samples.shape[:2])  # per sample, per batch
        for i in range(self.num_samples):
            # Put samples in prediction batch to get decisions and objective values
            batch_sample_i = samples[i]  # (B, num_parameters)
            for j in range(batch_sample_i.shape[0]):
                self.optmodel.setObj(batch_sample_i[j])
                sol, _ = self.optmodel.solve()

                obj = self.optmodel.cal_obj(costs[j], sol)
                objectives[i,j] = float(obj)
                
        # Compute loss function value
        # if self.loss_function_str == "regret":
        #     loss_terms = (objectives - optimal_objectives) * self.problem.opt_model.model_sense_int
        # elif self.loss_function_str == "objective":
        #     loss_terms = objectives * self.problem.opt_model.model_sense_int

        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss_terms: torch.Tensor =  (objectives - optimal_objectives) / optimal_objectives
        else:
            loss_terms: torch.Tensor = (optimal_objectives - objectives)/ optimal_objectives

        # loss_terms = loss_terms.mean(dim=0)  # take mean over samples
        # base_loss = loss_terms.detach().numpy().astype(np.float32)
        # loss_terms = loss_terms.float()

        if self.standardize_loss:
            loss_terms = self.standardize(loss_terms)

        # Compute surrogate loss for gradient
        base_loss = loss_terms.mean(dim=0).detach().numpy().astype(np.float32)
        loss = (loss_terms * log_probs).mean(dim=0)
        logger_loss = loss.detach().numpy().astype(np.float32)
        loss_mean = torch.mean(loss)

        # Update
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # Logging
        log_dict = {
            "loss": logger_loss,
            "eval": base_loss,
            "sigma": torch.sqrt(distribution.variance).detach().numpy().astype(np.float32),
        }
        return log_dict
    
    def run_batch(self, features: torch.Tensor, costs: torch.Tensor, optimal_objectives: torch.Tensor, metrics: list[str] | None = None) -> dict[str, torch.Tensor]:
        """
        Evaluates the predictor on a batch of data without updating the model.

        Args:
            features (torch.Tensor): The input features.
            costs (torch.Tensor): The costs associated with each sample.
            optimal_objectives (torch.Tensor): The optimal objectives.
            metrics (list[str] | None): List of additional metrics to evaluate. Defaults to None.
        """
        pred = self.noisifier.forward(features)

        # Get objective value per prediction
        objectives = torch.zeros(pred.shape[0]) 
        for i in range(pred.shape[0]):
            # Put samples in prediction batch to get decisions and objective values
            self.optmodel.setObj(pred[i])
            sol, _ = self.optmodel.solve()

            obj = self.optmodel.cal_obj(costs[i], sol)
            objectives[i] = float(obj)

        opt_obj_squeezed = optimal_objectives.squeeze()

        if self.optmodel.modelSense == EPO.MINIMIZE:
            loss_terms: torch.Tensor =  (objectives - opt_obj_squeezed) / opt_obj_squeezed
        else:
            loss_terms: torch.Tensor = (opt_obj_squeezed - objectives)/ opt_obj_squeezed

        loss = loss_terms.mean(dim=0)
        logger_loss = loss.detach().numpy().astype(np.float32)

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
        self.noisifier.train() if mode == "train" else self.noisifier.eval()

        # Update dist predictor t
        if self.noisifier.sigma_setting == "cooling":
            self.noisifier.update_t(epoch_num)

        # Initialize dictionary with the results
        epoch_results = []

        # Run
        for batch in data_loader:
            x, y, sol, obj = batch
            if mode == "train":
                batch_results = self.update(x, y, obj)
            else:
                batch_results = self.run_batch(x, y, obj)
            mode_batch_results = {f"{mode}/{key}": val for key, val in batch_results.items()}
            mode_batch_results["batch_size"] = len(x)
            epoch_results.append(mode_batch_results)

        return epoch_results
    
    def train_model(self, x_train, y_train, x_val, y_val):
        train_loader = torch.utils.data.DataLoader(
            optDataset(self.optmodel, x_train, y_train),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            optDataset(self.optmodel, x_val, y_val),
            batch_size=self.batch_size, shuffle=False
        )
        for epoch in range(self.num_epochs):
            _ = self.run_epoch("train", train_loader, epoch)
            val_results = self.run_epoch("validation", val_loader, epoch)

            # Calculate weighted average validation loss
            total_val_loss = sum(res["validation/loss"] * res["batch_size"] for res in val_results)
            total_val_samples = sum(res["batch_size"] for res in val_results)
            avg_val_loss = total_val_loss / total_val_samples

            # Early stopping check
            if self.early_stopper.step(avg_val_loss, self.noisifier):
                print(f"Early stopping at epoch {epoch} with validation loss {avg_val_loss:.4f}")
                break
        
        self.noisifier.eval()  # Set to evaluation mode after training
        return avg_val_loss
    
    def optimize(self, x, m = None): 
        # Predict
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        device = next(self.noisifier.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            pred_costs = self.noisifier.forward(x)

        
        self.optmodel.setObj(pred_costs)
        sol, obj = self.optmodel.solve()

        return sol, obj

    @staticmethod
    def standardize(batch: torch.Tensor, epsilon: float = 10**-5):
        """
        Standardizes a batch of losses along the batch dimension.

        Standardization serves as a variance-reduction baseline for the SFGE gradient estimator,
        based on insights from Silvestri et al. (2024).

        Args:
            batch (torch.Tensor): Batch losses to standardize.
            epsilon (float): Small value added to the standard deviation to avoid division by zero.
                Defaults to 1e-5.

        Returns:
            torch.Tensor: Standardized batch losses with zero mean and unit variance along dim 1.
        """
        # We standardize along the batch dimension (dim=1), using keepdim for broadcasting.
        mean_batch = torch.mean(batch, dim=1, keepdim=True)
        std_batch = torch.std(batch, dim=1, keepdim=True)

        # Broadcasting handles the element-wise operation correctly for both 1D and 2D cases.
        standardized = (batch - mean_batch) / (std_batch + epsilon)

        return standardized
