from abc import abstractmethod
from typing import Iterator

import torch
from torch import nn
import numpy as np
from torch.nn import Parameter


class Predictor(nn.Module):
    """
    Base class for all predictors. Predictors are restricted to be instances of torch.nn.Module.

    Attributes:
        num_inputs (int): The number of inputs for the predictor.
        num_outputs (int): The total number of outputs the predictor produces.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initializes the Predictor base class.

        Args:
            num_inputs (int): The number of inputs for the predictor.
            num_outputs (int): The total number of outputs the predictor produces.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the predictor. This method must be implemented by all subclasses.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the predictor.
        """
        raise NotImplementedError

    @staticmethod
    def _set_bias(layer: nn.Module, init_bias: float | np.ndarray) -> None:
        """
        Sets the bias of a given neural network layer. This method is static and can be called directly on the class.
        It handles both scalar and array-like initial bias values.

        Args:
            layer (nn.Module): The neural network layer whose bias is to be set.
                               This layer must have a 'bias' attribute.
            init_bias (float | np.ndarray): The bias value(s). Can be a float or a numpy array.
        """
        np_init_bias = np.array(init_bias)
        if len(np_init_bias.shape) == 0:  # Scalar bias
            if init_bias != 0.0:
                with torch.no_grad():
                    layer.bias.fill_(init_bias)
        else:  # Array bias
            layer.bias = nn.Parameter(torch.tensor(np_init_bias, dtype=torch.float32))





class MLPPredictor(Predictor, nn.Module):
    """
    A Multi-Layer Perceptron (MLP) predictor that extends both Predictor and nn.Module.
    This class constructs a feed-forward neural network with configurable layers,
    activation functions, and an optional scale-shift layer at the output.

    Attributes:
        mlp (nn.Sequential): The sequential neural network comprising the MLP layers.
        num_inputs (int): The number of input features for the predictor.
        num_outputs (int): The total number of output features the predictor produces.
        num_scenarios (int): The number of scenarios the predictor is designed to handle.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_layers: int = 2,  # 0 for a linear predictor
        hidden_dim: int = 256,
        dropout: float = 0.1,
        shared: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the MLPPredictor.

        Args:
            num_inputs (int): The number of input features.
            num_outputs (int): The number of output features.
            num_hidden_layers (int): The number of hidden layers in the MLP. 0 means a linear predictor. Defaults to 2.
            size (int): The number of neurons in each hidden layer. Defaults to 252.
            activation (Activation): The activation function to use for hidden layers.
                                     Can be a string (e.g., 'relu', 'leaky_relu') or an nn.Module.
                                     Defaults to 'leaky_relu'.
            output_activation (Activation): The activation function to use for the output layer.
                                            Can be a string or an nn.Module. Defaults to 'identity'.
            scale (float | np.ndarray): The scale factor for the output scale-shift layer. Defaults to 1.0.
            shift (float | np.ndarray): The shift factor for the output scale-shift layer. Defaults to 0.0.
            *args: Variable length argument list to be passed to nn.Module.__init__.
            **kwargs: Arbitrary keyword arguments to be passed to nn.Module.__init__.
        """
        if shared:
            num_outputs = 1

        Predictor.__init__(self, num_inputs, num_outputs)
        nn.Module.__init__(self, *args, **kwargs)


        layers = []
        in_size = num_inputs
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_dim

        # Apply output layer and output activation
        layers.append(nn.Linear(in_size, num_outputs))

        self.mlp = nn.Sequential(*layers)
        self.shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLPPredictor.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: The output tensor from the MLP.
        """
        output = self.mlp(x)
        if self.shared:
            output = output.squeeze(-1)
        return output

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over the module's parameters.

        Args:
            recurse (bool): If True, then yields parameters of this module
                            and all submodules. Otherwise, yields only parameters
                            that are direct members of this module. Defaults to True.

        Returns:
            Iterator[Parameter]: An iterator over the parameters.
        """
        return self.mlp.parameters(recurse=recurse)

    def get_first_layer(self) -> nn.Module:
        """
        Returns the first layer of the MLP. This can be used to adjust the weights of this layer, which is relevant
        when we are working with linear predictors for example (then it is the only layer).

        Returns:
            nn.Module: The first layer of the MLP.
        """
        return self.mlp[0]
