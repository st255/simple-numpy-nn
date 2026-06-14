import numpy as np
from Layers.Layer import Layer

class Tanh(Layer):
    def forward(self, inputs, training: bool = True):
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input