from __future__ import annotations

from abc import ABC
from enum import Enum

import numpy as np

from activation_functions import ActivationFunction


class TypeLayer(Enum):
    input_layer = "input"
    output_layer = "output"
    hidden_layer = "hidden"


class InterfaceLayer(ABC):
    activation_function: ActivationFunction = None
    neuron: int = None
    type_layer: TypeLayer = None
    weights: np.ndarray = None
    bias: np.ndarray = None

    def forward(self, **kwargs):
        pass

    def backward(self, **kwargs):
        pass


class AbstractLayer(InterfaceLayer):
    _next_layer: AbstractLayer = None
    _previous_layer: AbstractLayer = None

    def set_next(self, layer: AbstractLayer):
        self._next_layer = layer
        return layer

    def next(self) -> AbstractLayer:
        return self._next_layer

    def has_next(self) -> bool:
        return True if self._next_layer is not None else False

    def set_previous(self, layer: AbstractLayer):
        self._previous_layer = layer
        return layer

    def previous(self) -> AbstractLayer:
        return self._previous_layer

    def has_previous(self) -> bool:
        return True if self._previous_layer is not None else False

    def __str__(self):
        return self.type_layer.value


class Layer(AbstractLayer):
    def __init__(self, neuron: int, type_layer: TypeLayer, activation_function: ActivationFunction = None):
        self.forward_data = None
        self.output = None
        self.input = None
        self.activation_function = activation_function
        self.type_layer = type_layer
        self.neuron = neuron

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
