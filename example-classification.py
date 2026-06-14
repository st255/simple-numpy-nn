import numpy as np
from sklearn.datasets import make_moons

from Model import Model
from Layers.Linear import Linear
from Layers.ReLU import ReLU
from Layers.Sigmoid import Sigmoid
from Optimizers.GD import GD
from Loss.BCE import BCE


def train(epochs, model, optimizer, loss_fn, X, y):
    '''
    Auxiliary function to train the model
    '''
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
    print("Classification Example")
    print("===============================================================")

    # Classification data
    X_clf, y_clf = make_moons(n_samples=1000, noise=0.2, random_state=42)
    y_clf = y_clf.reshape(-1, 1)

    # Constructing the model
    model_clas = Model()
    model_clas.addLayer(Linear(2, 10))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Linear(10, 7))
    model_clas.addLayer(ReLU())    
    model_clas.addLayer(Linear(7, 1))
    model_clas.addLayer(Sigmoid())

    # Optimizer and loss function
    optim_clas = GD(model_clas, lr=0.01)
    loss_clas = BCE()

    # Training the model
    train(
        epochs=4000, 
        model=model_clas,
        optimizer=optim_clas,
        loss_fn=loss_clas,
        X=X_clf,
        y=y_clf
    )
    
    # Eval mode
    model_clas.training = False

    # Evaluation of the model
    preds_train = model_clas.forward(X_clf)
    preds_test = model_clas.forward(X_clf)
    
    # Calculamos métricas
    train_acc = np.mean((preds_train > 0.5).astype(int) == y_clf) * 100
    
    print(f"\nPrecisión en Entrenamiento: {train_acc:.2f}%")