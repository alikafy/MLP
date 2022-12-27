from __future__ import annotations

from abc import ABC

import numpy as np
from numpy import exp


class ActivationFunction(ABC):
    output = None
    input = None

    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented

    def forward(self, data):
        raise NotImplemented

    def backward(self, **kwargs):
        raise NotImplemented


class ConcreteActivationFunction(ActivationFunction):
    def __init__(self):
        self.output = None
        self.input = None

    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented

    def forward(self, input_data):
        self.input = input_data
        self.output = self.function(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.derivative(self.input) * output_error


class Linear(ConcreteActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(ConcreteActivationFunction):
    def function(self, x):
        return 1.0 / (1.0 + exp(-x))

    def derivative(self, x):
        return self.function(x) * (1.0 - self.function(x))


class Tanh(ConcreteActivationFunction):
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class Relu(ConcreteActivationFunction):
    def function(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return int(x > 0)
