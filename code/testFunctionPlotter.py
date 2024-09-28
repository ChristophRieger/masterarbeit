# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 17:05:07 2022

@author: chris
"""

import matplotlib.pyplot as plt
import numpy as np


# just plotting the double exp function
plt.close("all")
# Define the function y(x)
def y(x):  
  return (np.exp(-x/0.015) - np.exp(-x/0.001))
  # return np.e ** -x

# Generate x values from 0 to 1 with a step of 0.01
x = np.arange(-0.01, 0.05, 0.0001)

# Evaluate y(x) for each value of x
y_values = y(x)

i = 0
while x[i] < 0:
  y_values[i] = 0
  i = i + 1

# Plot the function
plt.plot(x, y_values)
plt.title("Kernel function", fontsize=14)
plt.xlabel("s", fontsize=15)
plt.xticks(fontsize=11)

plt.ylabel(r'$\varepsilon (s)$', fontsize=15)
plt.yticks(fontsize=11)

plt.show()