from abc import ABC
from enum import Enum


class ActivationFunction(ABC):
    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented


class Linear(ActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x):
        return 1


class TypeLayer(Enum):
    input_layer = "input"
    output_layer = "output"
    hidden_layer = "hidden"


class InterfaceLayer(ABC):
    activation_function: ActivationFunction = None
    neuron: int = None
    type_layer: TypeLayer = None


class Layer(InterfaceLayer):
    def __init__(self, neuron: int, type_layer: TypeLayer, activation_function: ActivationFunction = None):
        self.activation_function = activation_function
        self.type_layer = type_layer
        self.neuron = neuron
