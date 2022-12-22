from unittest import TestCase

from layer import Layer, TypeLayer
from activation_functions import Linear
from networks import Network
from perprocessing import PerProcessingBankData


class InitNetworkTest(TestCase):
    def setUp(self):
        pass

    def test_init_network(self):
        activation_function = Linear
        input_layer = Layer(neuron=3, type_layer=TypeLayer.input_layer)
        hidden_layer_1 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer, activation_function=activation_function())
        hidden_layer_2 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer, activation_function=activation_function())
        output_layer = Layer(neuron=2, type_layer=TypeLayer.output_layer, activation_function=activation_function())
        network = Network(input_layer=input_layer, hidden_layers=[hidden_layer_1, hidden_layer_2],
                          output_layer=output_layer, alpha=0.5)
        network.initial_network()
        initial_weights = network.initial_weights()
        self.assertEqual(input_layer.next(), hidden_layer_1)
        self.assertEqual(hidden_layer_1.next(), hidden_layer_2)
        self.assertEqual(hidden_layer_2.next(), output_layer)

    def test_read_csv_file(self):
        PerProcessingBankData("../data/BankWages2.csv").per_process()
