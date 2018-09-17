import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork

#df_train = pd.read_csv('devnagri_train.csv', header=None)
#df_test = pd.read_csv('devnagri_test_public.csv', header=None)
#
#X = np.array(df_train.iloc[:, 1:]).T
#X = X / 255
#X_test = np.array(df_test.iloc[:, 1:]).T
#X_test = X_test / 255
## print(X.shape, X_test.shape)
#
#Y = np.array(df_train.iloc[:, :1]).T
#Y_test = np.array(df_test.iloc[:, :1]).T
## One-hot encode the target
#num_classes = len(np.unique(Y))
## print(num_classes)
#Y = np.eye(num_classes)[Y.reshape(-1)].T
#Y_test = np.eye(num_classes)[Y_test.reshape(-1)].T
## print(Y.shape, Y_test.shape)

X = np.random.rand(5, 100)
Y = np.random.rand(3, 100)

num_inputs = X.shape[0]
num_outputs = Y.shape[0]  # = num_classes
hidden_archi = [4, 3, 2]

font_nn = NeuralNetwork(input_size = num_inputs,
                        output_size = num_outputs,
                        hidden_layer_sizes = hidden_archi,
                        activation = "relu")

font_nn.fit(X, Y, 0.01, 100, True, 16)

Ypred = font_nn.predict(X)






