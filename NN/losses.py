import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """The Loss function's implementation"""
        pass

    @abstractmethod
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """The derivative of the loss function's implementation"""
        pass


class MeanSquaredError(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return np.mean((y_true - y_pred) ** 2)

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return 2 * (y_pred - y_true) / np.size(y_true)

    def __repr__(self):
        return "MSE()"


class BinaryCrossEntropy(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return -np.sum(
            y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)
        )

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return (y_pred - y_true) / np.size(y_true)

    def __repr__(self):
        return "BCE()"


class CategoricalCrossEntropy(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> float:
        """Computes categorical cross-entropy loss."""
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        """Computes gradient of categorical cross-entropy loss."""
        return -y_true / (y_pred + 1e-9) / y_true.shape[0]

    def __repr__(self):
        return "CategoricalCrossEntropy()"
