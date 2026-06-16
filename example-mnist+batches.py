import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Model import Model
from Layers.Linear import Linear
from Layers.ReLU import ReLU
from Layers.Softmax import Softmax
from Optimizers.GD import GD
from Loss.CCE import CCE

def flatten_tensor_samples(tensor):
    '''
    Maps each sample (28x28) to 784 characteristics (1D array)
    '''
    num_samples = tensor.shape[0]
    flattened_tensor = tf.reshape(tensor, (num_samples, -1))
    return flattened_tensor.numpy()

def map_label(label):
    '''
    Aplies one-hot encoding to the label
    '''
    l = [0.0 for _ in range(10)]
    l[label] = 1.0
    return l

def train(epochs, model, optimizer, loss_fn, X, y, batch_size=256):
    '''
    Auxiliary function to train the model
    '''
    num_samples = X.shape[0]

    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))
        

        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            predictions_batch = model.forward(X_batch)
            loss_value = loss_fn.forward(predictions_batch, y_batch)
            loss_grad = loss_fn.backward(predictions_batch, y_batch)
            
            model.backward(loss_grad)
            optimizer.update()
            
            epoch_loss += loss_value
        
        # Loss per epoch
        epoch_loss /= num_batches
        
        print(f"Epoch {epoch:4d}/{epochs} | Loss Medio: {epoch_loss:.4f}")


if __name__ == "__main__":
    print("===============================================================")
    print("MNIST Example")
    print("===============================================================")

    # Classification data
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    X_train = flatten_tensor_samples(X_train_raw)
    X_test = flatten_tensor_samples(X_test_raw)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np.array(list(map(map_label, y_train_raw)))
    y_test = np.array(list(map(map_label, y_test_raw)))

    # Constructing the model
    model_clas = Model()
    model_clas.addLayer(Linear(784, 1024))
    model_clas.addLayer(ReLU())
    model_clas.addLayer(Linear(1024, 512))
    model_clas.addLayer(ReLU())    
    model_clas.addLayer(Linear(512, 10))
    model_clas.addLayer(Softmax())

    # Optimizer and loss function
    optim_clas = GD(model_clas, lr=0.01)
    loss_clas = CCE()

    # Training the model
    train(
        epochs=40, 
        model=model_clas,
        optimizer=optim_clas,
        loss_fn=loss_clas,
        X=X_train,
        y=y_train
    )
    
    # Eval mode
    model_clas.training = False

    # Evaluation of the model
    preds_train = model_clas.forward(X_train)
    preds_test = model_clas.forward(X_test)
    
    train_acc = np.mean((preds_train > 0.5).astype(int) == y_train) * 100
    
    print(f"\nTest acc: {train_acc:.2f}%")


    # Visualizing predictions
    pred_labels = np.argmax(preds_test, axis=1)
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test_raw[i], cmap='gray_r')
        
        prediccion = pred_labels[i]
        real = y_test_raw[i]
        
        color = 'green' if prediccion == real else 'red'
        
        ax.set_title(f"Pred: {prediccion}\nReal: {real}", fontsize=9, color=color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()