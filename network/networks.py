from abc import ABC
from typing import List

import numpy as np

from layer import AbstractLayer
from loss_functions import LostFunction


class InterfaceNetwork(ABC):
    hidden_layers: List[AbstractLayer] = None
    input_layer: AbstractLayer = None
    output_layer: AbstractLayer = None
    alpha: float = None

    def fit(self, **kwargs):
        pass

    def predict(self, data):
        pass

    def initial_network(self):
        pass

    def initial_weights(self):
        pass

    def forward(self, data, until_layer: AbstractLayer = None):
        pass

    def backward(self, **kwargs):
        pass


class Network(InterfaceNetwork):
    def __init__(self, input_layer: AbstractLayer, hidden_layers: List[AbstractLayer], output_layer: AbstractLayer,
                 alpha: float, loss_function: LostFunction):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.alpha = alpha
        self.loss_function = loss_function

    @property
    def layers(self):
        layers = []
        layer = self.input_layer.next()
        while layer:
            layers.append(layer)
            if layer.activation_function is not None:
                layers.append(layer.activation_function)
            layer = layer.next()
        return layers

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
            weight = np.random.rand(layer.previous().neuron, layer.neuron) - 0.5
            layer.weights = weight
            weights.append(weight)
        return weights

    def initial_bias(self):
        layers = []
        layers.extend(self.hidden_layers)
        layers.extend([self.output_layer])
        biases = []
        for layer in layers:
            bias = np.random.rand(1, layer.neuron) - 0.5
            layer.bias = bias
            biases.append(bias)
        return biases

    def forward(self, data, until_layer: AbstractLayer = None):
        if not self.input_layer.neuron == data.shape[1]:
            raise ValueError
        forward_data = data.copy()

        for layer in self.layers:
            if layer == until_layer:
                break
            forward_data = layer.forward(input_data=forward_data)

        return forward_data

    def fit(self, x_train, y_train, epochs, learning_rate):

        for epoch in range(epochs):
            total_loss = 0
            for index, data in enumerate(x_train):
                # forward propagation
                output = self.forward(data)
                # compute loss (for display purpose only)
                total_loss += self.loss_function.function(y_train[index], output)

                # backward propagation
                error = self.loss_function.derivative(y_train[index], output)

                for layer in reversed(self.layers):
                    error = layer.backward(output_error=error, learning_rate=learning_rate)
            # calculate average error on all samples
            total_loss /= len(x_train)
            print('epoch %d/%d   error=%f' % (epoch+1, epochs, total_loss))

    def predict(self, data):
        result = []
        for i in data:
            result.append(self.forward(i))
        return result
