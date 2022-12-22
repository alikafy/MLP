from abc import ABC
from typing import List

import numpy

from layer import Layer


class InterfaceNetwork(ABC):
    hidden_layers: List[Layer] = None
    input_layer: Layer = None
    output_layer: Layer = None
    alpha: float = None

    def train(self):
        pass

    def predict(self):
        pass


class Network(InterfaceNetwork):
    def __init__(self, input_layer: Layer, hidden_layers: List[Layer], output_layer: Layer, alpha: float):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.alpha = alpha

    def get_weights(self):
        return self.alpha

    def initial_network(self):
        before_layer = self.input_layer
        for layer in self.hidden_layers:
            before_layer.set_next(layer)
            before_layer = layer
        before_layer.set_next(self.output_layer)

    def initial_weights(self):
        layers = [self.input_layer]
        layers.extend(self.hidden_layers)
        weights = []
        for layer in layers:
            weight = numpy.random.rand(layer.neuron, layer.next().neuron)
            layer.weights = weight
            weights.append(weight)
        return weights
