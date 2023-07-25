import numpy as np

class PerceptronSimple: #regressão linear
    def __init__(self, learning_rate=0.01, i=100):
        self.lr = learning_rate
        self.iterations = i
        self.activation_func = self._activation_
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_features = X.shape[1]
        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        for elem in range(self.iterations):
            for idx, x_ in enumerate(X):
                linear_output = np.dot(x_, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y[idx]-y_predicted)
                self.weights += update * x_
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _activation_(self, x):
        """Sign"""
        return np.where(x>0, 1,0)