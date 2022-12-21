from abc import ABC
from typing import List

from network.layer import Layer


class InterfaceNetwork(ABC):
    hidden_layers: List[Layer] = None
    input_layer: Layer = None
    output_layer: Layer = None
    alpha: float = None
    weights = None


class Network(InterfaceNetwork):

    def __init__(self, input_layer: Layer, hidden_layers: List[Layer], output_layer: Layer, alpha: float):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.alpha = alpha
