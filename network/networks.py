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
    
    def initial_network(self):
        pass

    def initial_weights(self):
        pass

    def initial_bias(self):
        pass


class Network(InterfaceNetwork):
    def __init__(self, input_layer: Layer, hidden_layers: List[Layer], output_layer: Layer, alpha: float):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.alpha = alpha

    def initial_network(self):
        self.set_forward_flow_network()
        self.set_backward_flow_network()

    def set_forward_flow_network(self):
        before_layer = self.input_layer
        for layer in self.hidden_layers:
            before_layer.set_next(layer)
            before_layer = layer
        before_layer.set_next(self.output_layer)

    def set_backward_flow_network(self):
        next_layer = self.output_layer
        hidden_layers = self.hidden_layers.copy()
        hidden_layers.reverse()
        for layer in hidden_layers:
            next_layer.set_previous(layer)
            next_layer = layer
        next_layer.set_previous(self.input_layer)

    def initial_weights(self):
        layers = self.hidden_layers
        layers.extend([self.output_layer])
        weights = []
        for layer in layers:
            weight = numpy.random.rand(layer.previous().neuron, layer.neuron)
            layer.weights = weight
            weights.append(weight)
        return weights

    def initial_bias(self):
        layers = []
        layers.extend(self.hidden_layers)
        layers.extend([self.output_layer])
        biases = []
        for layer in layers:
            bias = numpy.random.rand(layer.neuron)
            layer.bias = bias
            biases.append(bias)
        return biases
