class Model:
    def __init__(self, training: bool = True):
        self.layers = []
        self.__training = training

    @property
    def training(self):
        return self.__training

    @training.setter
    def training(self, value):
        if not isinstance(value, bool):
            raise ValueError("Training must be a boolean value.")
        self.__training = value

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output, self.training)
        return output
    
    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
