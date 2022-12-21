from unittest import TestCase

from network.layer import Layer, TypeLayer, Linear
from network.network import Network


class InitNetworkTest(TestCase):
    def setUp(self):
        pass

    def test_init_network(self):
        activation_function = Linear
        input_layer = Layer(neuron=3, type_layer=TypeLayer.input_layer)
        output_layer = Layer(neuron=2, type_layer=TypeLayer.output_layer, activation_function=activation_function())
        hidden_layer_1 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer, activation_function=activation_function())
        hidden_layer_2 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer, activation_function=activation_function())
        network = Network(input_layer=input_layer, hidden_layers=[hidden_layer_1, hidden_layer_2],
                          output_layer=output_layer, alpha=0.5)
