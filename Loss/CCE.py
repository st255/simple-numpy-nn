import numpy as np
from Loss.Loss import Loss

class CCE(Loss):
    def forward(self, y_pred, y_true):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=-1)
        return np.mean(loss)

    def backward(self, y_pred, y_true):
        eps = 1e-8
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        grad = - (y_true / y_pred_clipped)
        return grad / y_true.shape[0]