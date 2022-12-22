from unittest import TestCase

from activation_functions import Linear
from layer import Layer, TypeLayer
from networks import Network
from perprocessing import PerProcessingBankData


class InitNetworkTest(TestCase):
    def setUp(self):
        activation_function = Linear
        self.input_layer = Layer(neuron=3, type_layer=TypeLayer.input_layer)
        self.hidden_layer_1 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer,
                                    activation_function=activation_function())
        self.hidden_layer_2 = Layer(neuron=4, type_layer=TypeLayer.hidden_layer,
                                    activation_function=activation_function())
        self.output_layer = Layer(neuron=2, type_layer=TypeLayer.output_layer,
                                  activation_function=activation_function())
        self.network = Network(input_layer=self.input_layer, hidden_layers=[self.hidden_layer_1, self.hidden_layer_2],
                               output_layer=self.output_layer, alpha=0.5)

    def test_init_network(self):
        self.network.initial_network()
        self.assertEqual(self.input_layer.next(), self.hidden_layer_1)
        self.assertEqual(self.hidden_layer_1.next(), self.hidden_layer_2)
        self.assertEqual(self.hidden_layer_2.next(), self.output_layer)

    def test_init_weights(self):
        self.network.initial_network()
        self.network.initial_weights()
        self.assertEqual(self.input_layer.weights.shape, (3, 4))
        self.assertEqual(self.hidden_layer_1.weights.shape, (4, 4))
        self.assertEqual(self.hidden_layer_2.weights.shape, (4, 2))
        self.assertEqual(self.output_layer.weights, None)

    def test_read_csv_file(self):
        PerProcessingBankData("../data/BankWages2.csv").per_process()
