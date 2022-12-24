import numpy as np

from error_functions import ErrorFunction
from layer import InterfaceLayer
from networks import Network


def forward(network: Network, data, until_layer: InterfaceLayer = None):
    # todo: i don't know when input size differed to input_layer size'
    if not network.input_layer.neuron == data.shape[0]:
        raise ValueError
    layer = network.input_layer
    forward_data = data.copy()
    while layer.has_next() and layer != until_layer:
        forward_data = np.dot(forward_data, layer.next().weights) + layer.next().bias
        layer = layer.next()
        # todo maybe function has other parameters like delta
        forward_data = layer.activation_function.function(forward_data)
    return forward_data


def backward(network: Network, alpha: float, error_function: ErrorFunction, data, target):
    # todo: learn network with batch and stochastic and sequential learning
    layer = network.output_layer

    prediction = forward(network, data)
    error = error_function.derivative(prediction=prediction, actual=target)
    delta = error * layer.activation_function.derivative(prediction)

    previous_weights = layer.weights

    while layer.has_previous():

        old_weights = layer.weights
        old_bias = layer.bias

        output_previous_layer = forward(network, data, layer.previous())
        new_weights = old_weights - alpha * np.kron(delta, output_previous_layer)
        new_bias = old_bias - alpha * delta
        delta = np.dot(delta, previous_weights) * layer.previous().activation_function.derivative(output_previous_layer)
        previous_weights = old_weights.copy()

        layer.weights = new_weights
        layer.bias = new_bias

        layer = layer.previous()
