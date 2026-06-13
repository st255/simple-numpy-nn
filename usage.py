import numpy as np
from sklearn.datasets import make_regression, make_moons

from Model import Model
from Layers.Linear import Linear
from Layers.ReLU import ReLU
from Layers.Sigmoid import Sigmoid
from Optimizers.GD import GD
from Loss.MSE import MSE
from Loss.BCE import BCE


def train(epochs, model, optimizer, loss_fn, X, y):
    for epoch in range(epochs):

        predictions = model.forward(X)
        loss_value = loss_fn.forward(predictions, y)
        loss_grad = loss_fn.backward(predictions, y)
        model.backward(loss_grad)
        
        optimizer.update()
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss_value:.4f}")

    return predictions


if __name__ == "__main__":
    print("===============================================================")
    print("Regression")
    print("===============================================================")

    X_reg, y_reg = make_regression(n_samples=5000, n_features=3, noise=10.0, random_state=42)
    y_reg = y_reg.reshape(-1, 1)

    model_reg = Model()
    model_reg.addLayer(Linear(3, 7))
    model_reg.addLayer(ReLU())
    model_reg.addLayer(Linear(7, 1))

    optim_reg = GD(model_reg, lr=0.001)
    loss_reg = MSE()

    train(
        epochs=3000,
        model=model_reg,
        optimizer=optim_reg,
        loss_fn=loss_reg,
        X=X_reg,
        y=y_reg
    )


    print("===============================================================")
    print("Classification")
    print("===============================================================")

    X_clf, y_clf = make_moons(n_samples=300, noise=0.15, random_state=42)
    y_clf = y_clf.reshape(-1, 1)

    model_clas = Model()
    model_clas.addLayer(Linear(2, 16))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Linear(16, 8))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Linear(8, 1))
    model_clas.addLayer(Sigmoid())


    optim_clas = GD(model_clas, lr=0.01)
    loss_clas = BCE()

    preds_clf = train(
        epochs=3000,
        model=model_clas,
        optimizer=optim_clas,
        loss_fn=loss_clas,
        X=X_clf,
        y=y_clf
    )
    
    bin_prec = (preds_clf > 0.5).astype(int)
    precision = np.mean(bin_prec == y_clf) * 100
    print(f"\nClas. model precision: {precision:.2f}%")