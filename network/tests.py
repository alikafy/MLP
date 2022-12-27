from unittest import TestCase

import numpy as np

from activation_functions import Linear, Sigmoid, Tanh
from layer import Layer, TypeLayer
from loss_functions import SSE, MSE
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
                               output_layer=self.output_layer, alpha=0.5, loss_function=SSE())

    def test_init_network(self):
        self.network.initial_network()
        self.assertEqual(self.input_layer.next(), self.hidden_layer_1)
        self.assertEqual(self.hidden_layer_1.next(), self.hidden_layer_2)
        self.assertEqual(self.hidden_layer_2.next(), self.output_layer)
        self.assertEqual(self.output_layer.next(), None)

        self.assertEqual(self.input_layer.previous(), None)
        self.assertEqual(self.hidden_layer_1.previous(), self.input_layer)
        self.assertEqual(self.hidden_layer_2.previous(), self.hidden_layer_1)
        self.assertEqual(self.output_layer.previous(), self.hidden_layer_2)

    def test_init_weights(self):
        self.network.initial_network()
        self.network.initial_weights()
        self.assertEqual(self.input_layer.weights, None)
        self.assertEqual(self.hidden_layer_1.weights.shape, (3, 4))
        self.assertEqual(self.hidden_layer_2.weights.shape, (4, 4))
        self.assertEqual(self.output_layer.weights.shape, (4, 2))

    def test_init_bias(self):
        self.network.initial_network()
        self.network.initial_weights()
        self.network.initial_bias()
        self.assertEqual(self.input_layer.bias, None)
        self.assertEqual(self.hidden_layer_1.bias.shape[0], 4)
        self.assertEqual(self.hidden_layer_2.bias.shape[0], 4)
        self.assertEqual(self.output_layer.bias.shape[0], 2)

    def test_read_csv_file(self):
        data = PerProcessingBankData("../data/BankWages2.csv").per_process()
        self.assertTrue(data is not None)


class TrainTest(TestCase):
    def setUp(self):
        activation_function = Tanh
        self.input_layer = Layer(neuron=2, type_layer=TypeLayer.input_layer)
        self.hidden_layer_1 = Layer(neuron=3, type_layer=TypeLayer.hidden_layer,
                                    activation_function=activation_function())
        self.hidden_layer_2 = Layer(neuron=3, type_layer=TypeLayer.hidden_layer,
                                    activation_function=activation_function())
        self.output_layer = Layer(neuron=1, type_layer=TypeLayer.output_layer,
                                  activation_function=activation_function())
        self.network = Network(input_layer=self.input_layer, hidden_layers=[self.hidden_layer_1, self.hidden_layer_2],
                               output_layer=self.output_layer, alpha=0.5, loss_function=MSE())
        self.network.initial_network()
        self.network.initial_weights()
        self.network.initial_bias()

    def test_forward_action(self):
        inputs, target = PerProcessingBankData("../data/BankWages2.csv").per_process()
        output = self.network.forward(inputs[0])
        self.assertEqual(output.shape[0], 1)

    def test_forward_action_with_until_layer(self):
        inputs, target = PerProcessingBankData("../data/BankWages2.csv").per_process()
        output = self.network.forward(inputs[0], self.input_layer)
        self.assertEqual(output.shape[0], 6)
        output = self.network.forward(inputs[0], self.hidden_layer_1)
        self.assertEqual(output.shape[0], 6)
        output = self.network.forward(inputs[0], self.hidden_layer_2)
        self.assertEqual(output.shape[0], 3)
        output = self.network.forward(inputs[0], self.output_layer)
        self.assertEqual(output.shape[0], 4)
        output = self.network.forward(inputs[0])
        self.assertEqual(output.shape[0], 1)

    def test_backward_action(self):
        inputs, target = PerProcessingBankData("../data/BankWages2.csv").per_process()
        x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
        self.network.fit(x_train=x_train, y_train=y_train, epochs=5000, learning_rate=0.1)
        predict = self.network.predict(x_train)
        self.assertTrue(predict[0] - y_train[0] < 0.1)
