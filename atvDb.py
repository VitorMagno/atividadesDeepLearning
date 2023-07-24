#%%
import numpy as np
from perceptrons import PerceptronSimple
# %%
X_train = np.load("./db/X_train.npy")
X_test = np.load("./db/X_test.npy")
y_train = np.load("./db/y_train.npy")
y_test = np.load("./db/y_test.npy")
#%%
def accuracy(predictions, y_test):
    classifications = len(y_test)
    correct_classifications = sum(p == r for p, r in zip(predictions, y_test))
    return correct_classifications / classifications
# %%
alfa = 0.0001
i = 10000
perc = Perceptron(alfa, i)
# %%
perc.fit(X_train, y_train)
predictions = perc.predict(X_test)
print(f"Perceptron classification accuracy {accuracy(y_test, predictions)*100}%")
# %%
