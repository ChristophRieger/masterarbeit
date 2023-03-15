# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:12:47 2022

@author: chris
"""
import numpy as np

# Use a double exponential Kernel
# t: current simulation time
# tf: spike timing
# tauRise: time constant for rising part of function
# tauDecay: time constant for falling part of function
# returns: EPSP Ytilde
def YTilde(t, dt, tf, tauRise, tauDecay):
  # !!! i added +dt here, because otherwise tf can be bigger than t, which should not happen
  # this means i use the upper boundary of the current timestep as the current time
  return np.exp(-(t+dt-tf)/tauDecay) - np.exp(-(t+dt-tf)/tauRise)

# Use a double exponential Kernel
# t: current simulation time
# tf: spike timing
# tauRise: time constant for rising part of function
# tauDecay: time constant for falling part of function
# returns: EPSP Ytilde
def ZTilde(t, dt, tf, tauRise, tauDecay):
  # !!! i added +dt here, because otherwise tf can be bigger than t, which should not happen
  # this means i use the upper boundary of the current timestep as the current time
  return np.exp(-(t+dt-tf)/tauDecay) - np.exp(-(t+dt-tf)/tauRise)