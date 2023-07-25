#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
#%%
np.random.seed(1)
X, y = make_blobs(n_samples=150,n_features=2,centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
#%%
class SimplePerceptron:
    
    def __init__(self, alfa=0.01, epoch=100):
        self.bias = 1
        self.weights = None
        self.alfa = alfa
        self.epoch = epoch

    def _feedfoward(self, X, y_train):
        n_features = len(X)
        self.weights = np.random.rand(n_features)
        self.bias = 1
        for elem in range(self.epoch):
            for index, x in enumerate(X):
                y_predicted = self.weightedSum(X[index], self.weights[index], self.bias)
                if(self.activation(y_predicted)) == 0:
                    self.weights[index] = self.weights[index] + self.update(self.alfa, y_train[index], y_predicted)
                    self.bias = self.weights[index] + self.update(self.alfa, y_train[index], y_predicted)
    
    def _update(alfa, y_train, y_predicted, x):
        return ((alfa * (y_train - y_predicted)) * x)
    
    def weightedSum(value, w, bias):
        return np.dot(value, w) + bias
    
    def activation(self, value):
        # signal
        # if value <=0:
        #     return 0
        # else:
        #     return 1
        
        # ReLu
        # if value < 0:
        #     return value
        # else:
        #     return 0
        
        """sigmoid"""
        return 1/ (np.exp(-value)+1)
    
    def softmax(values):
        expo = np.exp(values)
        sum_expo = np.sum(np.exp(values))
        return expo/sum_expo
    
    def predict(self, X_test, y_test):
        test = (X_test * self.weights) + self.bias
        result = self.activation(test)
        return self.evaluate(result, y_test)
    
    def evaluate(self, result, y_test):
        return np.sum(y_test == result) / len(y_test)
# %%
smpPcpt = SimplePerceptron(0.01, 100)
smpPcpt.train(X_train, y_train)
smpPcpt.predict(X_test, y_test)