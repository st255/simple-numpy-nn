import numpy as np
from Loss.Loss import Loss

class BCE(Loss):
    def forward(self, y_pred, y_true):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    def backward(self, y_pred, y_true):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        grad = ((1 - y_true) / (1 - y_pred_clipped) - y_true / y_pred_clipped)
        return grad / y_true.size