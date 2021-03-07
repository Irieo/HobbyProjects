# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:20:36 2021

@author: iegor
"""

import numpy as np

inputs = [1, 2, 3]
weights = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

#biases = [0.0, 0.0, 0.0]

layer_outputs = np.dot(weights,inputs)

layer1a = np.dot(weights,inputs)
layer1b = np.dot(inputs, weights)

print(layer_outputs)
print(layer1a)
print(layer1b)

for i in range(len(layer_outputs)):
    x = np.dot(weights[i],inputs)
    print(x)

for i in range(len(layer_outputs)):
    y = np.dot(inputs[i],weights)
    print(y)