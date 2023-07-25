#%%
from neuralNetwork import activation
import numpy as np
# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,n_features=2,centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
# %%
class Perceptron_linear():
    def __init__(self):
        self.weights = None
        self.bias = None

    def perceptron_linear_training(self, x, y, alfa, i):
        self.weights = np.random.rand(x.shape[1])
        self.bias = np.random.rand(x.shape[1])

        for elem in range(i):
            for index in range(0, x.shape[1]):
                # feed forward
                result = self.weightedSum(x[index], self.weights[index]) + self.bias
                # output = activation function
                output = self.activation(result)
                # # # adjust weights
                erro = y[index] - output[index]
                print(erro)
                if erro != 0:
                    if (index+1) == x.shape[1]:
                        self.weights[0] += self.weights[index] + self.update(alfa, erro)
                        self.bias[0] += self.bias[index] + self.update(alfa, erro)
                    else:
                        self.weights[index+1] += self.weights[index] + self.update(alfa, erro)
                        self.bias[index+1] += self.bias[index] + self.update(alfa, erro)

    def weightedSum(self, x, w):
        return np.dot(x, w)

    def update(self, alfa, erro):
        return (alfa*(erro))

    def activation(self, x):
        return np.where(x>0, 1,0)

    def predict(self, x):
        result = x * self.weights + self.bias
        output = activation(result)
        return output

    def evaluate_precision(self,y_predicted, y_real):
        return np.sum(y_real == y_predicted)/len(y_predicted)
# %%
percp = Perceptron_linear()
percp.perceptron_linear_training(X_train, y_train, 0.01, 100)
predictions = percp.predict(X_test)
result = percp.evaluate_precision(predictions, y_test)
print(result)
# %%
