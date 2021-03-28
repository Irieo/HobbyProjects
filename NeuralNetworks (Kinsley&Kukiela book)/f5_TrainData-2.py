# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:44:24 2021

@author: iegor
"""

import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
        
X, y = spiral_data(samples=5, classes=2)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take 3 outputs) and 3 output values
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()   

# Make a forward pass of our training data through this layer
dense1.forward(X)
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
dense2.forward(activation1.output)
activation2.forward(dense2.output)
        

print(dense1.output[:]) 
print('----activation----')
print(activation1.output[:])

print(dense2.output[:]) 
print('----activation----')
print(activation2.output[:])


