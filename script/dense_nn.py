

# TODO: loads weights and dataset, implement the model and report accuracy


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Read MNIST Dataset in from CSV file
print('Reading CSV Data...')
data = pd.read_csv('./data/mnist_train.csv')

# Preprocess Data
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

#normalize X
X = X / 255.0


# Load weights + Biases
W1 = np.loadtxt('data/model/weights_hidden.csv', delimiter=',') #h*n
W2 = np.loadtxt('data/model/weights_output.csv', delimiter=',') #m*h
b1 = np.loadtxt('data/model/biases_hidden.csv', delimiter=',') #h
b2 = np.loadtxt('data/model/biases_output.csv', delimiter=',') #m

W1T = np.transpose(W1)
W2T = np.transpose(W2)


def tanh(x):
    return np.tanh(x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X):
    H = tanh(X @ W1T +b1)
    Z = sigmoid(H @ W2T + b2)
    predY = np.argmax(Z, axis=1)
    return predY
# measure accuracy on test set using forward propagation for all MNIST data

predY = forward(X)
accuracy = np.mean(predY == y)
print(f'Accuracy: {accuracy * 100:.2f}%')

# optionally visualize some of the results using matplotlib

for i in range(5):
    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {predY[i]}, True: {y[i]}")
    plt.savefig(f"digit_{i}.png")
    plt.show()