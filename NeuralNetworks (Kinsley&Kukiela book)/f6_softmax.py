# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:02:35 2021

@author: iegor
"""


import numpy as np
np.set_printoptions(suppress=True)

inputs = np.array([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])

class Activation_Softmax:

    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
        
        
activation = Activation_Softmax()    

print(activation.forward(inputs))
