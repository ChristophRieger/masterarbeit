# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:10:15 2022

@author: chris
"""
import numpy as np

# =============================================================================
# Updates the Weights between all Y Neurons and the one Z Neuron that fired
# =============================================================================
def updateWeights(YTilde, weights, ZNeuronID, c, learningRate):
  # !!! YTilde is max 0.8 in this current implementation. if c is set to 1 (altough at
  # other points the say its e**-5 which makes it even worse), as said in paper
  # this means, that deltaWeights can only be positive, if the weight is negative.
  # this seems very wrong to me, as weights should never get negative
  deltaWeights = learningRate * (YTilde * c * np.exp(-weights[:, ZNeuronID]) - 1)
  weights[:, ZNeuronID] += deltaWeights
  # !!! disabled clip, as with c = 10 weights should stay within 0 and 1 !!!
  # weights = np.clip(weights, 0.001, 5)
  return weights

def updateIntrinsicWeights(intrinsicWeights, ZNeuronID, c, learningRate):
  Z = np.zeros(len(intrinsicWeights))
  Z[ZNeuronID] = 1
  intrinsicWeights += learningRate * (c* np.exp(-intrinsicWeights) * Z - 1)
  # intrinsicWeights = np.clip(intrinsicWeights, 0.001, 5)
  return intrinsicWeights

def activateInhibition():
  
  return True










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
  return np.exp(-x/0.015) - np.exp(-x/0.001)

# Generate x values from 0 to 1 with a step of 0.01
x = np.arange(0, 0.05, 0.0001)

# Evaluate y(x) for each value of x
y_values = y(x)

# Plot the function
# plt.plot(x, y_values)
plt.show()