from forward_propagations import forward_propagate


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
