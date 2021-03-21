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
        
        
X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

activation1 = Activation_ReLU()

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
dense1.forward(X)

activation1.forward(dense1.output)

print(dense1.output[:5]) 
print(activation1.output[:5])