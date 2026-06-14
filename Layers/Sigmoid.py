import numpy as np
from Layers.Layer import Layer

class Sigmoid(Layer):
    def forward(self, inputs, training: bool = True):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input
