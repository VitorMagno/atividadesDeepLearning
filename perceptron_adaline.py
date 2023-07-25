#%%
import enum
from neuralNetwork import activation
import numpy as np
# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
X_train = np.load("./db1/X_train.npy")
X_test = np.load("./db1/X_test.npy")
y_train = np.load("./db1/y_train.npy")
y_test = np.load("./db1/y_test.npy")
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.show()
#%%
w = None
# %%
def neuronio_unidade_linear(x,y, alfa, i):
    w = np.random.rand(x.shape[1])

    for elem in range(i):
        for indx, h in enumerate(x):
            # feed forward
            result = np.dot(x[indx,:],w[indx,:])
            print(result)
            # output = activation function
            output = activation(result)
            # adjust weights
            erro = y - output
            if erro != 0:
                w += update(alfa, erro)
    return w

def update(alfa, erro):
    return (alfa*(erro))

def activation(x):
    return 1/(1+np.exp(-x))

def predict(x):
    result = np.dot(x, w)
    return result

def evaluate_precision(y_predicted, y_real):
    return ((y_real - y_predicted) / len(y_predicted))
# %%
