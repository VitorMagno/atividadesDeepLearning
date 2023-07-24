#%%
import numpy as np
from perceptrons import PerceptronBetter
# %%
X_train = np.load("./db1/X_train.npy")
X_test = np.load("./db1/X_test.npy")
y_train = np.load("./db1/y_train.npy")
y_test = np.load("./db1/y_test.npy")
#%%
def accuracy(predictions, y_test):
    classifications = len(y_test)
    correct_classifications = sum(p == r for p, r in zip(predictions, y_test))
    return correct_classifications / classifications
# %%
alfa = 0.1
i = 1000
perc = Perceptron(alfa, i)
# %%
# 1
perc.fit(X_train, y_train)
predictions = perc.predict(X_test)
print(f"Perceptron classification accuracy {accuracy(y_test, predictions)*100}%")
