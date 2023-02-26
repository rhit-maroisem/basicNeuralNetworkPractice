# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# gets data from CSV file and turns it to NP array
data = pd.read_csv('datasets/train.csv')
data = np.array(data)

# get m and n values from array dimensions, then shuffle for random data
m, n = data.shape
np.random.shuffle(data)

# splits data into training and testing groups
data_test = data[0:1000].T  # .T transposes the array, kinda flipping it (refer back to notes)
Y_test = data_test[0]
X_test = data_test[1:n]  # this is why we found n

data_train = data[1000:m].T  # m is max number of images
Y_train = data_train[0]
X_train = data_train[1:n]


# initializing the parameters (weights and biases)
def init_params():
    # params for first layer
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    # params for second layer
    w2 = np.random.rand(10, 784) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


# ReLu activation function, makes things more interesting
def ReLU(z):
    return np.maximum(0, z)


# softmax activation function, gives us an actually percent probability of which number is is
def softmax(z):
    top = np.exp(z)
    bottom = np.sum(np.exp(z))
    return top / bottom


# forward propagation function: gets a prediction based on input data
def forward_prop(w1, b1, w2, b2, X):
    # first layer (unactivated)
    z1 = w1.dot(X) + b1
    # activating first layer
    a1 = ReLU(z1)

    # second layer (unactivated)
    z2 = w2.dot(a1) + b2
    # activating second layer
    a2 = softmax(z2)
    return a2


# one hot encoding the return values for back propagation
def one_hot(Y):
    one_hot_y = np.zeroes((Y.size(), Y.max()+1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


# derivative of ReLU function, necessary for backwards propagation
# returns slope, so 1 if element is positive, 0 if element is 0
def deriv_ReLU(z):
    return z > 0


# backwards propagation function: finds error (propagation) of each weight and bias
def back_prop(z1, a1, z2, a2, w2, X, Y):
    # define m
    m = Y.size
    # gets a one hot encoded matrix
    one_hot_y = one_hot(Y)

    # error in second layer
    dz2 = a2 - one_hot_y
    dw2 = (1 / m) * dz2.dot(a1.T)
    db2 = (1 / m) * np.sum(dz2)

    # error in first layer
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = (1 / m) * dz1.dot(X.T)
    db1 = (1 / m) * np.sum(dz1)

    return dw1, db1, dw2, db2


# updating parameters of network based on backwards propagation
def update_params(w1, b1, w2, b2,dw1, db1, dw2, db2, alpha):
    #updates params
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2






