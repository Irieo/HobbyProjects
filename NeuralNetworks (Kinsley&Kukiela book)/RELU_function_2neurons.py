# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:00:04 2021

@author: iegor
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def ACT(x):
    return np.maximum(0, x)

def layer1(x): 
        return ACT(-1*x + 0.5)

def layer2(x):
        return ACT(-2*x + 1.0)

def layer12(x):
        return ACT(-2*layer1(x) + 1.0)

x = []
y1 = []
y2 = []
y3 = []

for i in np.arange(-5, 5, 0.01):
    x.append(i) 
    y1.append(layer1(i))
    y2.append(layer2(i))
    y3.append(layer12(i))
               
#s1 = plt.scatter(x,y1)
#s2 = plt.scatter(x,y2)
s12 = plt.scatter(x,y3)
