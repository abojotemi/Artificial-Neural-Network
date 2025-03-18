import numpy as np
from numpy import ndarray

def xavier_init(input_dim: int, output_dim: int) -> ndarray:
    limit = np.sqrt(6 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

@staticmethod
def he_init(input_dim: int, output_dim: int) -> ndarray:
    stddev = np.sqrt(2 / input_dim)
    return np.random.normal(0, stddev, size=(input_dim, output_dim))
