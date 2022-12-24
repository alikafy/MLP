from __future__ import annotations

from abc import ABC
from math import tanh

from numpy import exp


class ActivationFunction(ABC):
    def function(self, x):
        raise NotImplemented

    def derivative(self, x):
        raise NotImplemented


class Linear(ActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(ActivationFunction):
    def function(self, x):
        return 1.0 / (1.0 + exp(-x))

    def derivative(self, x):
        return self.function(x) * (1.0 - self.function(x))


class Tanh(ActivationFunction):
    def function(self, x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    def derivative(self, x):
        return 1 - tanh(x) ** 2


class Relu(ActivationFunction):
    def function(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return int(x > 0)
