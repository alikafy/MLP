from __future__ import annotations

from abc import ABC


class ErrorFunction(ABC):
    def function(self, prediction, actual):
        raise NotImplemented

    def derivative(self, prediction, actual):
        raise NotImplemented


class SSE(ErrorFunction):
    def function(self, prediction, actual):
        return 0.5 * (prediction - actual) ** 2

    def derivative(self, prediction, actual):
        return prediction - actual
