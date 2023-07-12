# %%
import numpy as np
import matplotlib.pyplot as plt
# %%

from sklearn.datasets import make_blobs

X, y = make_blobs(centers=2, cluster_std=1, random_state=1)

plt.scatter(X[:,0], X[:,1], c=y)

plt.show()
#%%
def step(z):
    if z > 0:
        return 1
    else:
        return 0

#%%
weights = np.random.rand(3)
taxa = 0.01

#%%

def perf(y, y_chapeu):
    return (1/2)*(y - y_chapeu)

##for 
entrada = np.append(X[0],1)

# Neuronio
z = weights * entrada
z = np.sum(z)

# Ativação
z = step(z)

loss = perf(y[0], z)
z

weights = weights * (taxa * loss * entrada)

#%%
y[0]
