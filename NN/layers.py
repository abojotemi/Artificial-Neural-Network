import numpy as np
from numpy import ndarray
from scipy.special import expit
from typing import Callable
import NN.weight_initializer as weight_initializer
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, _input: ndarray, train=True) -> ndarray:
        """Forward propagation method"""
        pass

    def backward(self, prev_grad: ndarray) -> ndarray:
        """Backward propagation method"""
        pass


class Activation(Layer):
    def __init__(
        self, func: Callable[[ndarray], ndarray], grad: Callable[[ndarray], ndarray]
    ):
        self.func = func
        self.grad = grad

    def forward(self, _input: ndarray, train=True) -> ndarray:
        if train:
            self.input = _input
        return self.func(_input)

    def backward(self, prev_grad: ndarray) -> ndarray:
        return prev_grad * self.grad(self.input)


class ReLU(Activation):
    def __init__(self):
        def func(x):
            return np.maximum(0, x)

        def grad(x):
            return np.where(x > 0, 1, 0)

        super().__init__(func, grad)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Activation):
    def __init__(self):
        def func(x):
            return expit(x)

        def grad(x):
            return func(x) * (1 - func(x))

        super().__init__(func, grad)

    def __repr__(self):
        return "Sigmoid()"


class Softmax(Activation):
    def __init__(self):
        def stable_softmax(x):
            shifted = x - np.max(x, axis=-1, keepdims=True)
            exps = np.exp(shifted)
            return exps / np.sum(exps, axis=-1, keepdims=True)

        def grad(x):
            return x 

        super().__init__(stable_softmax, grad)

    def backward(self, prev_grad: np.ndarray) -> np.ndarray:
        s = self.func(self.input)
        if s.ndim == 1:
            jacobian = np.diag(s) - np.outer(s, s)
            return jacobian.dot(prev_grad)
        else:

            grad_input = np.empty_like(s)
            for i in range(s.shape[0]):
                jacobian = np.diag(s[i]) - np.outer(s[i], s[i])
                grad_input[i] = jacobian.dot(prev_grad[i])
            return grad_input

    def __repr__(self):
        return "Softmax()"


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        initializer: Callable[[int, int], ndarray] = weight_initializer.he_init,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initializer(input_size, output_size)
        self.bias = np.zeros((1, output_size))
        self.grad_weights: ndarray | None = None
        self.grad_bias: ndarray | None = None

    def forward(self, _input: ndarray, train=True) -> ndarray:
        if train:
            self.input = _input
        return _input.dot(self.weights) + self.bias

    def backward(self, prev_grad: ndarray) -> ndarray:
        out_grad = prev_grad.dot(self.weights.T)
        self.grad_weights = self.input.T.dot(prev_grad)
        self.grad_bias = np.sum(prev_grad, axis=0, keepdims=True)
        return out_grad

    def __repr__(self):
        return f"Dense(input={self.input_size}, output={self.output_size})"
