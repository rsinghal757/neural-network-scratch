import numpy as np

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0, Z) 
    return A

def tanh(Z):
    a, b = np.exp(Z), np.exp(-Z)
    A = (a - b) / (a + b)
    return A

def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, Z):
    t = tanh(Z)
    dZ = dA * (1-t) * (1+t)
    return dZ