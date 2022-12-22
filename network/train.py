import numpy as np

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
