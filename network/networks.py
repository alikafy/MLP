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

    def train(self):
        pass

    def predict(self):
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
            weight = np.random.rand(layer.previous().neuron, layer.neuron)
            layer.weights = weight
            weights.append(weight)
        return weights

    def initial_bias(self):
        layers = []
        layers.extend(self.hidden_layers)
        layers.extend([self.output_layer])
        biases = []
        for layer in layers:
            bias = np.random.rand(layer.neuron)
            layer.bias = bias
            biases.append(bias)
        return biases

    def forward(self, data, until_layer: AbstractLayer = None):
        if not self.input_layer.neuron == data.shape[0]:
            raise ValueError
        layer = self.input_layer
        forward_data = data.copy()
        while layer and layer != until_layer:
            forward_data = layer.forward(input_data=forward_data)
            layer = layer.next()

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

                layer = self.output_layer
                while layer.has_previous():
                    error = layer.backward(output_error=error, learning_rate=learning_rate)
                    layer = layer.previous()

            # calculate average error on all samples
            total_loss /= len(x_train)
            print('epoch %d/%d   error=%f' % (epoch+1, epochs, total_loss))
