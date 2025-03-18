from numpy import ndarray
import numpy as np
from tqdm import tqdm

from NN.optimizers import Optimizer


def train(nn, X: ndarray, y: ndarray, optimizer: Optimizer, epochs: int = 1000, batch_size: int = 32) -> ndarray:
    inputs,targets = X,y
    N = inputs.shape[0]
    if len(targets.shape) == 1:
        targets = targets[:,np.newaxis]
    loss_per_epochs = np.zeros(epochs)
    for e in tqdm(range(epochs)):
        batch_idx = np.random.permutation(N)
        batch_iter = (N // batch_size) + (1 if N % batch_size else 0)
        cumm_loss = 0
        for b in range(batch_iter):
            batch = batch_idx[b * batch_size: (b+1) * batch_size]
            batch_inputs, batch_targets = inputs[batch,:], targets[batch,:]
            batch_output = nn.forward(batch_inputs)
            cumm_loss += nn.loss.base(batch_targets, batch_output)
            gradient = nn.loss.grad(batch_targets, batch_output)
            nn.backward(gradient)
            optimizer.step()
        loss_per_epochs[e] = cumm_loss / batch_iter
        if e % 2 == 0:
            print(f"Epoch: {e} \t Loss: {cumm_loss}")
    return loss_per_epochs
    
def __call__(nn, _input: ndarray) -> ndarray:
    return nn.forward(_input)