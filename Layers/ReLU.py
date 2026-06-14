import numpy as np
from Layers.Layer import Layer

class ReLU(Layer):
    def forward(self, inputs, training: bool = True):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input
