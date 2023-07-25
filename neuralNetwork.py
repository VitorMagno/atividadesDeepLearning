#%%
import numpy as np

#%%
X_train = np.load("./db1/X_train.npy")
X_test = np.load("./db1/X_test.npy")
y_train = np.load("./db1/y_train.npy")
y_test = np.load("./db1/y_test.npy")
# %%
bias_hI = None
bias_hO = None
weights_h_input = None
weights_h_output = None
# %%
input_size = X_train.shape[1]
hidden_size = 2
output_size = 1
weights_h_input = np.random.rand(input_size, hidden_size)
weights_h_output = np.random.rand(hidden_size, output_size)
bias_hI = np.random.rand(hidden_size)
bias_hO = np.random.rand(output_size)
#%%
def train(X,y, epoch, alfa):
     for elem in range(epoch):
        input_layer_out,output_layer_out = feedForward(X)
        # calculate the error
        backPropagation(input_layer_out, output_layer_out)
        error = output - y
        delta = error * derivative(output)
        print(delta)

def activation(output):
    return 1/(1+np.exp(-output))

def derivative(output):
    return output*(1-output)

def feedForward(X):
    hidden_layer_input = np.dot(X, weights_h_input) + bias_hI
    hidden_layer_output = activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_h_output) + bias_hO
    output_layer_output = activation(output_layer_input)
    backPropagation(output_layer_input,output_layer_output)
    return output_layer_output

def backPropagation():
    pass
#%%
train(X_train, y_test, epoch=100, alfa=0.01)
# %%
