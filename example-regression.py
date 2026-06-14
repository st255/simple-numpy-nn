from sklearn.datasets import make_regression

from Model import Model
from Layers.Linear import Linear
from Layers.ReLU import ReLU
from Optimizers.GD import GD
from Loss.MSE import MSE


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
    print("Regression Example")
    print("===============================================================")

    # Regression data
    X_reg, y_reg = make_regression(n_samples=5000, n_features=3, noise=10.0, random_state=42)
    y_reg = y_reg.reshape(-1, 1)

    # Constructing the model
    model_reg = Model()
    model_reg.addLayer(Linear(3, 7))
    model_reg.addLayer(ReLU())
    model_reg.addLayer(Linear(7, 1))

    # Optimizer and loss function
    optim_reg = GD(model_reg, lr=0.001)
    loss_reg = MSE()

    # Training the model
    train(
        epochs=3000,
        model=model_reg,
        optimizer=optim_reg,
        loss_fn=loss_reg,
        X=X_reg,
        y=y_reg
    )