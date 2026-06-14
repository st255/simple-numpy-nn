import numpy as np
from Layers.Layer import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs, training: bool = True):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output):
        self.dweights = np.dot(self.inputs.T, grad_output)
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

