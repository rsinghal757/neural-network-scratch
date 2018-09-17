import random
import math
import numpy as np
from Activations import relu, relu_backward, sigmoid, sigmoid_backward, tanh, tanh_backward

def forward_step(A_prev, W, b, activation):
    
    if activation == "relu":
        
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b, Z)
        A = relu(Z)
        
    elif activation == "tanh":
        
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b, Z)
        A = tanh(Z)  
    
    elif activation == "sigmoid":
        
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b, Z)
        A = sigmoid(Z)

    return A, cache

def forward_propagation(X, parameters, activation):

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A 
        A, cache = forward_step(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation=activation)
        caches.append(cache)

    AL, cache = forward_step(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL, Y):

    m = Y.shape[1]  # No. of training samples
    cost = (1./ (2 * m)) * np.sum(np.multiply((AL - Y), (AL - Y)))
    cost = np.squeeze(cost)
    
    return cost

def gradient_calc(dZ, A_prev, W, b):

    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T) 
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_step(dA, cache, activation):

    A_prev, W, b, Z = cache
    
    if activation == "relu":

        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = gradient_calc(dZ, A_prev, W, b)
        
    elif activation == "tanh":
        
        dZ = tanh_backward(dA, Z)
        dA_prev, dW, db = gradient_calc(dZ, A_prev, W, b)
        
    elif activation == "sigmoid":
        
        dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = gradient_calc(dZ, A_prev, W, b)
    
    return dA_prev, dW, db

def backward_propagation(AL, Y, caches, activation):
    
    grads = {}
    L = len(caches)  # No. of layers (excluding input layer)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = np.divide((AL - Y), m)

    cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = backward_step(dAL, cache, activation="sigmoid")
    
    for l in reversed(range(L-1)):
        
        cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = backward_step(grads["dA" + str(l + 2)], cache, activation=activation)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2  # No. of layers (excluding input layer)

    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters

def generate_mini_batches(X, Y, mini_batch_size = 32, seed = 0):

    np.random.seed(seed)
    m = X.shape[1]
    num_classes = Y.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((num_classes, m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


class NeuralNetwork(object):
    
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation):
        
        self.parameters = {}
        self.activation = activation
        layer_dims = [input_size] + hidden_layer_sizes + [output_size]
        L = len(layer_dims)  # No. of layers (including input layer)
    
        for l in range(1, L):
            
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
    def fit(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False, batch_size=32):

        self.costs = []
        
        for i in range(0, num_iterations):
            
            t = i + 1
            learning_rate = learning_rate / math.sqrt(t)
            seed = random.randint(1, 1000)
            minibatches = generate_mini_batches(X, Y, batch_size, seed)
            
            for minibatch in minibatches:
                
                (batch_X, batch_Y) = minibatch
            
                AL, caches = forward_propagation(batch_X, self.parameters, self.activation)
                cost = compute_cost(AL, batch_Y)
                grads = backward_propagation(AL, batch_Y, caches, self.activation)
                self.parameters = update_parameters(self.parameters, grads, learning_rate)
                
            if print_cost == True and i % 10 == 0:
                self.costs.append(cost)
                print("Iteration: " + str(i), "Cost: " + str(cost))
                
    def predict(self, X):
        
        AL, caches = forward_propagation(X, self.parameters, self.activation)
        
        return AL
        
                
    











