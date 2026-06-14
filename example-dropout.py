import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from Model import Model
from Layers.Linear import Linear
from Layers.ReLU import ReLU
from Layers.Sigmoid import Sigmoid
from Layers.Dropout import Dropout
from Optimizers.GD import GD
from Loss.BCE import BCE


DROPOUT_RATE = 0.1 # Change this value to see the effect of dropout in the model


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
    print("Dropout Example")
    print("===============================================================")

    # Classification data, we use a small dataset to force overfitting
    X_clf, y_clf = make_moons(n_samples=10, noise=0.3, random_state=42)
    y_clf = y_clf.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    # Constructing the model, we made it bigger to force overfitting
    model_clas = Model()
    model_clas.addLayer(Linear(2, 1024))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Dropout(DROPOUT_RATE))
    
    model_clas.addLayer(Linear(1024, 512))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Dropout(DROPOUT_RATE))
    
    model_clas.addLayer(Linear(512, 1))
    model_clas.addLayer(Sigmoid())

    # Optimizer and loss function
    optim_clas = GD(model_clas, lr=0.1)
    loss_clas = BCE()

    # Training the model
    train(
        epochs=4000, 
        model=model_clas,
        optimizer=optim_clas,
        loss_fn=loss_clas,
        X=X_train,
        y=y_train
    )
    
    # Eval mode, we set training=False to disable dropout during evaluation
    model_clas.training = False

    # Evaluation of the model on both training and test sets
    preds_train = model_clas.forward(X_train)
    preds_test = model_clas.forward(X_test)
    
    train_acc = np.mean((preds_train > 0.5).astype(int) == y_train) * 100
    test_acc = np.mean((preds_test > 0.5).astype(int) == y_test) * 100
    
    print(f"\nTraining acc: {train_acc:.2f}%")
    print(f"Test acc: {test_acc:.2f}%")