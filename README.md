# Simple numpy neural network
---
#### Disclaimer
This project was developed as a learning and research exercise to understand better how neural networks work. Feel free to use the code as you wish.

## Summary

A simple implementation of different parts of a feed-forward neural network using numpy, including:

- Layers
- Activation functions
- Loss functions
- Forward pass
- Backpropagation
- Optimizers

No machine learning frameworks such as PyTorch are used.

## How it works
The following formulas are used to calculate the results. The input tensor must have a size (batch_size, characteristics).

$$
Z^{(t)} = A^{(t-1)} W^{(t)} + b^{(t)}
$$

$$
A^{(t)} = \sigma\!\left(Z^{(t)}\right)
$$

## Implemented Features

- Layers
  - [x] Dense layers
  - [x] Dropout
- Activation functions
  - [x] ReLU activation
  - [x] Sigmoid activation
  - [x] Tanh
  - [x] Softmax activation
- Loss functions
  - [x] MSE
  - [x] BCE
  - [x] Cross-entropy loss
- Optimizers
  - [x] Gradient descent optimizer
  - [ ] Adam optimizer
- Others
  - [x] MNIST example
  - [ ] Benchmark
  - [ ] Test units

## Results
The following model has been tested in the MNIST dataset using mini-batch gradient descent for 40 epochs, reaching an accuracy >= 99%.

```python
model_clas = Model()
model_clas.addLayer(Linear(784, 1024))
model_clas.addLayer(ReLU())
model_clas.addLayer(Linear(1024, 512))
model_clas.addLayer(ReLU())    
model_clas.addLayer(Linear(512, 10))
model_clas.addLayer(Softmax())
```

```
Test acc: 99.35%
```

![MNIST results](./mnist_results.png)


## Example of use
- `example-classification.py`: full-batch gradient descent for a simple classification problem. 
- `example-regression.py`: full-batch gradient descent for a simple regression problem.
- `example-dropout.py`: full-batch gradient descent forcing overfitting. Change the dropout rate to see how it affects the acurracy.
- `example-mnist+batches.py`: mini-batch gradient descent for the mnist dataset.
