from NN.layers import Layer, Dense
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, layers: list[Layer], lr: float):
        self.layers: list[Dense] = [layer for layer in layers if isinstance(layer, Dense)]
        self.lr = lr
    
    @abstractmethod
    def step(self, gradient: ndarray):
        raise NotImplementedError("Optimizer Step function not implemented") 


class SGD(Optimizer):
    def __init__(self, layers: list[Layer], lr: float = 0.1, momentum: float = 0.0):
        self.momentum = momentum
        self.velocity = {}
        super().__init__(layers, lr)
    
    def step(self):
        for idx, layer in enumerate(self.layers):
            if idx not in self.velocity:
                self.velocity[idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }
            self.velocity[idx]['weights'] = self.momentum * self.velocity[idx]['weights'] - self.lr * layer.grad_weights
            self.velocity[idx]['bias'] = self.momentum * self.velocity[idx]['bias'] - self.lr * layer.grad_bias
            layer.weights += self.velocity[idx]['weights']
            layer.bias += self.velocity[idx]['bias']
            
    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum})"
    
class RMSProp(Optimizer):
    def __init__(self, layers: list[Layer], lr: float = 0.1, beta = 0.9, epsilon: float = 1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}
        super().__init__(layers, lr)
        
    def step(self):
        for idx, layer in enumerate(self.layers):
            if idx not in self.cache:
                self.cache[idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }
            self.cache[idx]['weights'] = (self.beta * self.cache[idx]['weights'] + (1 - self.beta) * (layer.grad_weights ** 2))
            self.cache[idx]['bias'] = (self.beta * self.cache[idx]['bias'] + (1 - self.beta) * (layer.grad_bias ** 2))
            layer.weights -= self.lr * layer.grad_weights  / (np.sqrt(self.cache[idx]['weights'] ) + self.epsilon)
            layer.bias -= self.lr * layer.grad_bias  / (np.sqrt(self.cache[idx]['bias'] ) + self.epsilon)
            
    def __repr__(self):
        return f"RMSProp(lr={self.lr}, beta={self.beta})"

    
class Adam(Optimizer):
    def __init__(self, layers: list[Layer], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} 
        self.v = {} 
        self.count = 0  
        super().__init__(layers, lr)
    
    def step(self):
        self.count += 1
        for idx, layer in enumerate(self.layers):
            if idx not in self.m:
                self.m[idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }
            if idx not in self.v:
                self.v[idx] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }

            
            self.m[idx]['weights'] = (self.beta1 * self.m[idx]['weights'] + (1 - self.beta1) * layer.grad_weights)
            self.m[idx]['bias'] = (self.beta1 * self.m[idx]['bias'] + (1 - self.beta1) * layer.grad_bias)
            
            
            self.v[idx]['weights'] = (self.beta2 * self.v[idx]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2))
            self.v[idx]['bias'] = (self.beta2 * self.v[idx]['bias'] + (1 - self.beta2) * (layer.grad_bias ** 2))

            m_hat_weights = self.m[idx]['weights'] / (1 - self.beta1 ** self.count)
            m_hat_bias = self.m[idx]['bias'] / (1 - self.beta1 ** self.count)

            v_hat_weights = self.v[idx]['weights'] / (1 - self.beta2 ** self.count)
            v_hat_bias = self.v[idx]['bias'] / (1 - self.beta2 ** self.count)

            layer.weights -= self.lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
            layer.bias -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
    
    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"
