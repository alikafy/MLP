from __future__ import annotations

from abc import ABC

import numpy as np


class LostFunction(ABC):
    def function(self, prediction, actual):
        raise NotImplemented

    def derivative(self, prediction, actual):
        raise NotImplemented


class SSE(LostFunction):
    def function(self, prediction, actual):
        return 0.5 * (prediction - actual) ** 2

    def derivative(self, prediction, actual):
        return prediction - actual


class MSE(LostFunction):
    def function(self, prediction, actual):
        return np.mean(np.power(actual - prediction, 2))

    def derivative(self, prediction, actual):
        return 2 * (prediction - actual) / actual.size
