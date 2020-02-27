#!/usr/bin/env python
# coding: utf-8

# In[4]:


from math import e
import numpy as np
import pandas as pd
from numpy.random import rand

def sigmoid(x):
    return 1/(1 + np.power(e , (-x)))

def sigmidDerivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def checkThetaAndX(X, Y, Theta):
    num_of_layers = Theta.shape[0]
    #TODO and pass to logistic regression
    
    #Check if amount of corresponds equals to a first matrix of parameter's dimension
    #If it doesn't include bias we should add one to the amount of features
    bias_account = 1
    if(X.shape[0] + bias_account != Theta[0].shape[1]):
        return False
    
    #Check if each next matrix will have corresponding shape to the previous matrix
    for i in np.arange(num_of_layers - 1):
        if Theta[i].shape[0]  + bias_account != Theta[i + 1].shape[1]:
            return False
    #Check if number of classes corresponds to the shape of the last matrix
    if(Y.shape[0] != Theta[num_of_layers - 1].shape[0]):
        return False
    return True

def read_data(fileName, labels):
    k = len(labels)
    Data = pd.read_csv(fileName)
    data = np.array(Data).transpose()[:-2]

    m = data.shape[1]
    num_of_features = data.shape[0]
    y = np.array(Data).transpose()[-1:]

    Y = np.zeros([k, y.shape[1]])
    for i in range(y.shape[1]):
        for j in range(k):
            if y[0, i] == labels[j]:
                Y[j, i] = 1
    X=np.array(data, dtype = np.float64)
    return X, Y

def gen_Theta(input_layer_size, output_layer_size, hidden_layer_sizes, eps = 0.01):
    Theta = []
    sizes = np.concatenate(([input_layer_size], hidden_layer_sizes, [output_layer_size]))
    for i in np.arange(1, len(sizes)):
        Theta.append((rand(sizes[i], sizes[i-1] + 1) - 0.5)*eps)
    return np.array(Theta)

def gradient_descent(function_and_gradient, start_point, alpha=1e-1, eps=1e-10, max_iter=50):
    def f(x):
        return function_and_gradient(x)[0]
    def f_deriv(x):
        return function_and_gradient(x)[1]
    point = np.array(start_point)
    prev_value = f(point) - 100
    counter = 0
    while( counter < max_iter and abs(f(point) - prev_value) > eps):
        prev_value = f(point)
        point = point - alpha*f_deriv(point)
        counter += 1
    return point

def forward_propagation(X, Theta):
    Layers = []
    num_of_features = X.shape[0]
    m = X.shape[1]
    num_of_layers = Theta.shape[0]
    curLayer = X
    Layers.append(X)
    for i in range(Theta.shape[0]):
        curLayer = np.vstack([np.ones([1,m]), curLayer])
        curLayer = sigmoid(Theta[i].dot( curLayer ) )
        Layers.append(curLayer)
    return Layers




def J(X, Y, Theta):
    num_of_features = X.shape[0]
    m = X.shape[1]
    num_of_layers = Theta.shape[0]
    #TODO and pass to logistic regression
    if not checkThetaAndX(X, Y, Theta):
        print("You've enntered wrong data")
        return 0
    #If it doesn't include bias we should add one to the amount of features
    H = forward_propagation(X, Theta)[-1]
    return (np.sum(np.sum(-Y*np.log(H)-(1-Y)*np.log(1-H))) * (1/m))





