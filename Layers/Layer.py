from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, inputs, training: bool = True):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass

