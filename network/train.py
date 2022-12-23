import numpy as np

from error_functions import ErrorFunction
from networks import Network


def forward(network: Network, data):
    # todo: i don't know when input size differed to input_layer size'
    if not network.input_layer.neuron == data.shape[0]:
        raise ValueError
    layer = network.input_layer
    forward_data = data.copy()
    while layer.has_next():
        forward_data = np.dot(forward_data, layer.weights)
        layer = layer.next()
        # todo maybe function has other parameters like delta
        forward_data = layer.activation_function.function(forward_data)
    return forward_data


def backward(network: Network, alpha: float, error_function: ErrorFunction, data, target):
    # todo: learn network with batch and stochastic and sequential learning
    prediction = forward(network, data)
    error = error_function.derivative(prediction=prediction, actual=target)
    layer = network.input_layer
    while layer.has_next():
        old_weights = layer.weights
        derivative_error_weight = 1  # todo: i don't know how calculate this value because is different for every layer
        new_weights = old_weights - alpha * derivative_error_weight
        layer.weights = new_weights
        layer = layer.next()
