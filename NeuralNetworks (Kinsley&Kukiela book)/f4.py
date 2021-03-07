# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 09:29:45 2021

@author: iegor
"""

import numpy as np
a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a]).T
b = np.array([b])

print(np.dot(b,a))