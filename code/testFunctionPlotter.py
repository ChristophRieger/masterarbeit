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
  # c needs to be dependent on w which are constantly changing or very big.. paper lies that c > 1 is sufficient, its wrong
  c = 5
  # the middle term goes up to about 0.8
  # return (c * np.exp(-0.5)) * (np.exp(-x/0.015) - np.exp(-x/0.001)) - 1
  return np.e ** -x

# Generate x values from 0 to 1 with a step of 0.01
x = np.arange(1, 5, 0.0001)

# Evaluate y(x) for each value of x
y_values = y(x)

# Plot the function
plt.plot(x, y_values)
plt.show()