from __future__ import annotations

from abc import ABC, abstractmethod
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

    @abstractmethod
    def set_next(self, layer: InterfaceLayer):
        pass

    @abstractmethod
    def next(self) -> InterfaceLayer:
        pass


class AbstractLayer(InterfaceLayer):
    _next_layer: InterfaceLayer = None

    def set_next(self, layer: InterfaceLayer):
        self._next_layer = layer
        return layer

    def next(self) -> InterfaceLayer:
        return self._next_layer


class Layer(AbstractLayer):
    def __init__(self, neuron: int, type_layer: TypeLayer, activation_function: ActivationFunction = None):
        self.activation_function = activation_function
        self.type_layer = type_layer
        self.neuron = neuron

    def activate(self, input_data: float):
        return self.activation_function.function(input_data)
