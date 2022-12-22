from __future__ import annotations

from abc import ABC


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
