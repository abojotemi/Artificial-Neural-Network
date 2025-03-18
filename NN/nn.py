from numpy import ndarray
from contextlib import contextmanager
from NN.layers import Layer
from NN.losses import Loss
from NN.losses import MeanSquaredError

class ANN:
    def __init__(self, layers: list[Layer], loss: Loss = MeanSquaredError ):
        self.layers = layers
        self.training = True
        self.loss: Loss = loss()
    def forward(self, _input: ndarray) -> ndarray:
        output = _input
        for layer in self.layers:
            output = layer.forward(output, self.training)
        return output
    
    def backward(self, grad: ndarray) -> ndarray:
        gradient = grad
        for layer in self.layers[::-1]:
            gradient = layer.backward(gradient)
    @contextmanager
    def no_grad(self):
        self.training = False
        yield
        self.training = True
    def __call__(self, _input: ndarray) -> ndarray:
        return self.forward(_input)

    def __repr__(self):
        return str(self.layers)         
                