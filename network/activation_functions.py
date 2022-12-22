from __future__ import annotations

from abc import ABC

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
        return x * (1.0 - x)
