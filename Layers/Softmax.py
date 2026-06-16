import numpy as np
from Layers.Layer import Layer

class Softmax(Layer):
    def forward(self, inputs, training: bool = True):
        self.inputs = inputs
        exp_inputs = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        sum_grad_out = np.sum(grad_output * self.output, axis=-1, keepdims=True)
        grad_input = self.output * (grad_output - sum_grad_out)
        return grad_input