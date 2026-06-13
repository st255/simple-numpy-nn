from abc import ABC, abstractmethod
from Model import Model

class Optimizer(ABC):
    def __init__(self, model: Model, lr: float = 0.01):
        self.model = model
        self.lr = lr

    @abstractmethod
    def update(self):
        pass