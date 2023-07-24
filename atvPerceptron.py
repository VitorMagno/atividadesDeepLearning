# %%
import numpy as np
from perceptrons import PerceptronSimple
# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,n_features=2,centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# %%
alfa = 0.01
i = 1000
percep = Perceptron(alfa, i)

# %%
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# %%
percep.fit(X_train, y_train)
predictions = percep.predict(X_test)
print(f"Perceptron classification accuracy {accuracy(y_test, predictions)*100}%")

# %%
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)

x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

x1_1 = (-percep.weights[0] *  x0_1 - percep.bias)/percep.weights[1]
x1_2 = (-percep.weights[0] *  x0_2 - percep.bias)/percep.weights[1]

ax.plot([x0_1,x0_2],[x1_1,x1_2], 'k')

ymin = np.amin(X_train[:,1])
ymax = np.amax(X_train[:,1])
ax.set_ylim([ymin-3,ymax+3])

plt.show()
# %%