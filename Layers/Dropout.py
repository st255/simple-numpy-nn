import numpy as np
from Layers.Layer import Layer

class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        self.p = p

    def forward(self, inputs, training: bool = True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, size = inputs.shape) / (1 - self.p)
            return inputs * self.mask
        return inputs

    def backward(self, grad_output):
        return grad_output * self.mask


