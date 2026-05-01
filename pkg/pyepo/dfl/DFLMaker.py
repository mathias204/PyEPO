from abc import ABC, abstractmethod

class DFLMaker(ABC):
   
    @abstractmethod
    def train_model(self, x_train, y_train, x_val, y_val):
        """
        Train the DFL model using the provided training and validation data.

        Args:
            x_train (torch.Tensor): Training features.
            y_train (torch.Tensor): Training labels.
            x_val (torch.Tensor): Validation features.
            y_val (torch.Tensor): Validation labels.
        Returns:
            float: The final validation loss after training.
        """
        pass

    @abstractmethod
    def optimize(self, x, m = None):
        """
        Optimize the decision based on the input features.

        Args:
            x (torch.Tensor): Input features for optimization.
            m (optional): Additional model or parameters for optimization. Defaults to None.

        Returns:
            The optimized decision or output based on the input features.
        """
        pass