from Optimizers.Optimizer import Optimizer
from Model import Model

class GD(Optimizer):
    def __init__(self, model: Model, lr: float = 0.01):
        super().__init__(model, lr)

    def update(self):
        for layer in self.model.layers:
            if hasattr(layer, 'weights') and layer.dweights is not None:
                layer.weights -= self.lr * layer.dweights
                layer.biases -= self.lr * layer.dbiases