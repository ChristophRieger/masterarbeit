# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:55:32 2022

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')
import dataGenerator
import poissonGenerator
import neuronFunctions
import kernel

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import os
import pickle
import mathematischeAnalyse
import copy

# Command Center
loadWeights = False
numberOfRuns = 1

useCustomValues = True
if useCustomValues:
  # TODO Here I inject my values for the simulation output probabilities
  analysisProb = []
  simulProb = []
  simulStd = []
  
  # Empty Template
  ########################################
  # customFInput = 
  # customFPrior = 
  # customTauDecay = 
  # customC = False
  # customEnablePrior = False
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6]
  # barYLim = [0, 0.75]
  # # 1
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # # 2
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # # 3
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # # 4
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # # 5
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # # 6
  # analysisProb.append([])
  # simulProb.append([])
  # simulStd.append([])
  # customKLD = 
  # customKLDStd = 
  ########################################
  
  # Empty Template, with size 9 and DISABLED prior
  ########################################
  # customFInput = 
  # customFPrior = 
  # customTauDecay = 
  # customC = False
  # customEnablePrior = False
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6]
  # barYLim = [0, 0.75]
  # # 1
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([])
  # simulStd.append([])
  # # 2
  # analysisProb.append([0.143, 0.518, 0.304, 0.035])
  # simulProb.append([])
  # simulStd.append([])
  # # 3
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([])
  # simulStd.append([])
  # # 4
  # analysisProb.append([0.291, 0.613, 0.048, 0.048])
  # simulProb.append([])
  # simulStd.append([])
  # # 5
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([])
  # simulStd.append([])
  # # 6
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([])
  # simulStd.append([])
  # customKLD = 
  # customKLDStd = 
  ########################################
  
  # Empty Template, with size 9 and ENABLED prior
  ########################################
  # customFInput = 
  # customFPrior = 
  # customTauDecay = 
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([])
  # simulStd.append([])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([])
  # simulStd.append([])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([])
  # simulStd.append([])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([])
  # simulStd.append([])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([])
  # simulStd.append([])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([])
  # simulStd.append([])
  # customKLD = 
  # customKLDStd = 
  ########################################
  
  
  
  
  
  ########################################
  # customFInput = 42
  # customFPrior = 0
  # customTauDecay = 15
  # customC = False
  # customEnablePrior = False
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6]
  # barYLim = [0, 0.75]
  # # 1
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.124, 0.699, 0.121, 0.056])
  # simulStd.append([0.0061, 0.0093, 0.0058, 0.0033])
  # # 2
  # analysisProb.append([0.143, 0.518, 0.304, 0.035])
  # simulProb.append([0.104, 0.587, 0.266, 0.043])
  # simulStd.append([0.0059, 0.0090, 0.0067, 0.0025])
  # # 3
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.430, 0.421, 0.070, 0.080])
  # simulStd.append([0.0089, 0.0073, 0.0045, 0.0042])
  # # 4
  # analysisProb.append([0.291, 0.613, 0.048, 0.048])
  # simulProb.append([0.245, 0.553, 0.090, 0.111])
  # simulStd.append([0.0065, 0.0072, 0.0045, 0.0053])
  # # 5
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.124, 0.697, 0.123, 0.056])
  # simulStd.append([0.0046, 0.0069, 0.0050, 0.0043])
  # # 6
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.428, 0.421, 0.070, 0.081])
  # simulStd.append([0.0116, 0.0111, 0.0034, 0.0047])
  # customKLD = 0.0241
  # customKLDStd = 0.00159
  ########################################

  ########################################
  # customFInput = 70
  # customFPrior = 0
  # customTauDecay = 15 
  # customC = False
  # customEnablePrior = False
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8]
  # barYLim = [0, 0.95]
  # # 1
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.061, 0.861, 0.061, 0.017])
  # simulStd.append([0.0038, 0.0065, 0.0041, 0.0016])
  # # 2
  # analysisProb.append([0.143, 0.518, 0.304, 0.035])
  # simulProb.append([0.049, 0.715, 0.226, 0.011])
  # simulStd.append([0.0027, 0.0083, 0.0070, 0.0017])
  # # 3
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.472, 0.464, 0.027, 0.037])
  # simulStd.append([0.0100, 0.0101, 0.0027, 0.0032])
  # # 4
  # analysisProb.append([0.291, 0.613, 0.048, 0.048])
  # simulProb.append([0.207, 0.685, 0.043, 0.065])
  # simulStd.append([0.0067, 0.0080, 0.0037, 0.0042])
  # # 5
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.061, 0.864, 0.057, 0.018])
  # simulStd.append([0.0033, 0.0062, 0.0036, 0.0022])
  # # 6
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.475, 0.459, 0.028, 0.038])
  # simulStd.append([0.0086, 0.0087, 0.0018, 0.0032])
  # customKLD = 0.0915
  # customKLDStd = 0.004
  ########################################
  
  ########################################
  # customFInput = 88
  # customFPrior = 0
  # customTauDecay = 4 
  # customC = False
  # customEnablePrior = False
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6]
  # barYLim = [0, 0.75]
  # # 1
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.126, 0.695, 0.124, 0.054])
  # simulStd.append([0.0050, 0.0066, 0.0049, 0.0039])
  # # 2
  # analysisProb.append([0.143, 0.518, 0.304, 0.035])
  # simulProb.append([0.106, 0.587, 0.264, 0.043])
  # simulStd.append([0.0054, 0.0098, 0.0072, 0.0028])
  # # 3
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.429, 0.421, 0.070, 0.080])
  # simulStd.append([0.0080, 0.0083, 0.0037, 0.0044])
  # # 4
  # analysisProb.append([0.291, 0.613, 0.048, 0.048])
  # simulProb.append([0.240, 0.556, 0.095, 0.109])
  # simulStd.append([0.0044, 0.0084, 0.0036, 0.0058])
  # # 5
  # analysisProb.append([0.175, 0.612, 0.175, 0.038])
  # simulProb.append([0.125, 0.697, 0.124, 0.054])
  # simulStd.append([0.0044, 0.0072, 0.0038, 0.0032])
  # # 6
  # analysisProb.append([0.453, 0.453, 0.047, 0.047])
  # simulProb.append([0.426, 0.424, 0.072, 0.078])
  # simulStd.append([0.0085, 0.0068, 0.0031, 0.0047])
  # customKLD = 0.0233
  # customKLDStd = 0.00145
  ########################################
  
  ########################################
  # customFInput = 42
  # customFPrior = 222
  # customTauDecay = 15
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([0.015, 0.962, 0.015, 0.008])
  # simulStd.append([0.0021, 0.0030, 0.0017, 0.0016])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([0.013, 0.938, 0.042, 0.007])
  # simulStd.append([0.0016, 0.0040, 0.0032, 0.0013])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([0.084, 0.891, 0.010, 0.015])
  # simulStd.append([0.0038, 0.0046, 0.0018, 0.0016])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([0.039, 0.931, 0.012, 0.018])
  # simulStd.append([0.0033, 0.0045, 0.0017, 0.0024])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([0.050, 0.321, 0.043, 0.587])
  # simulStd.append([0.0033, 0.0087, 0.0039, 0.0105])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([0.175, 0.172, 0.021, 0.632])
  # simulStd.append([0.0077, 0.0106, 0.0018, 0.0158])
  # customKLD = 0.0188
  # customKLDStd = 0.00145
  ########################################
  
  ########################################
  # customFInput = 88
  # customFPrior = 440
  # customTauDecay = 4
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([0.011, 0.974, 0.010, 0.005])
  # simulStd.append([0.0016, 0.0026, 0.0019, 0.0013])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([0.010, 0.957, 0.028, 0.005])
  # simulStd.append([0.0017, 0.0034, 0.0036, 0.0011])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([0.066, 0.915, 0.009, 0.011])
  # simulStd.append([0.0031, 0.0044, 0.0014, 0.0018])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([0.026, 0.952, 0.009, 0.012])
  # simulStd.append([0.0030, 0.0039, 0.0017, 0.0020])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([0.052, 0.328, 0.046, 0.573])
  # simulStd.append([0.0027, 0.0085, 0.0031, 0.0089])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([0.172, 0.174, 0.025, 0.630])
  # simulStd.append([0.0061, 0.0059, 0.0025, 0.0095])
  # customKLD = 0.0112
  # customKLDStd = 0.00088
  ########################################
  
  ########################################
  # customFInput = 88
  # customFPrior = 600
  # customTauDecay = 4
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([0.003, 0.993, 0.003, 0.001])
  # simulStd.append([0.0011, 0.0013, 0.0008, 0.0005])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([0.003, 0.987, 0.009, 0.001])
  # simulStd.append([0.0009, 0.0024, 0.0019, 0.0008])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([0.023, 0.971, 0.003, 0.003])
  # simulStd.append([0.0025, 0.0030, 0.0009, 0.0011])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([0.008, 0.984, 0.003, 0.004])
  # simulStd.append([0.0011, 0.0019, 0.0008, 0.0011])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([0.028, 0.182, 0.023, 0.766])
  # simulStd.append([0.0026, 0.0077, 0.0025, 0.0079])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([0.089, 0.088, 0.010, 0.813])
  # simulStd.append([0.0057, 0.0047, 0.0016, 0.0081])
  # customKLD = 0.0615
  # customKLDStd = 0.00235
  ########################################
  
  ########################################
  # customFInput = 98
  # customFPrior = 440
  # customTauDecay = 4
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = False
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([0.009, 0.979, 0.009, 0.004])
  # simulStd.append([0.0016, 0.0020, 0.0010, 0.0009])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([0.009, 0.963, 0.025, 0.003])
  # simulStd.append([0.0015, 0.0023, 0.0020, 0.0008])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([0.067, 0.916, 0.007, 0.009])
  # simulStd.append([0.0051, 0.0051, 0.0012, 0.0019])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([0.025, 0.955, 0.009, 0.011])
  # simulStd.append([0.0031, 0.0033, 0.0012, 0.0019])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([0.051, 0.379, 0.046, 0.523])
  # simulStd.append([0.0035, 0.0084, 0.0027, 0.0084])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([0.191, 0.188, 0.022, 0.600])
  # simulStd.append([0.0080, 0.0061, 0.0021, 0.0116])
  # customKLD = 0.0101
  # customKLDStd = 0.0009
  ########################################
  
  ########################################
  # customFInput = 98
  # customFPrior = 440
  # customTauDecay = 4
  # customC = False
  # customEnablePrior = True
  # customDoubleSize = True
  # barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  # barYLim = [0, 1]
  # # 1
  # analysisProb.append([0.010, 0.977, 0.010, 0.002])
  # simulProb.append([0.002, 0.996, 0.002, 0.000])
  # simulStd.append([0.0008, 0.0011, 0.0007, 0.0002])
  # # 2
  # analysisProb.append([0.010, 0.967, 0.021, 0.002])
  # simulProb.append([0.002, 0.982, 0.016, 0.000])
  # simulStd.append([0.0007, 0.0026, 0.0025, 0.0002])
  # # 3
  # analysisProb.append([0.035, 0.957, 0.004, 0.004])
  # simulProb.append([0.089, 0.907, 0.001, 0.002])
  # simulStd.append([0.0058, 0.0059, 0.0004, 0.0009])
  # # 4
  # analysisProb.append([0.017, 0.977, 0.003, 0.003])
  # simulProb.append([0.016, 0.979, 0.002, 0.003])
  # simulStd.append([0.0021, 0.0023, 0.0006, 0.0010])
  # # 5
  # analysisProb.append([0.088, 0.308, 0.088, 0.515])
  # simulProb.append([0.022, 0.799, 0.019, 0.160])
  # simulStd.append([0.0025, 0.0071, 0.0018, 0.0066])
  # # 6
  # analysisProb.append([0.205, 0.205, 0.021, 0.570])
  # simulProb.append([0.356, 0.352, 0.007, 0.285])
  # simulStd.append([0.0088, 0.0100, 0.0013, 0.0081])
  # customKLD = 0.1411
  # customKLDStd = 0
  ########################################
  
  ########################################
  customFInput = 98
  customFPrior = 440
  customTauDecay = 4
  customC = 3
  customEnablePrior = True
  customDoubleSize = False
  barYTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  barYLim = [0, 1]
  # 1
  analysisProb.append([0.010, 0.977, 0.010, 0.002])
  simulProb.append([0.004, 0.988, 0.005, 0.003])
  simulStd.append([0.0010, 0.0016, 0.0011, 0.0008])
  # 2
  analysisProb.append([0.010, 0.967, 0.021, 0.002])
  simulProb.append([0.005, 0.977, 0.014, 0.004])
  simulStd.append([0.0009, 0.0028, 0.0018, 0.0012])
  # 3
  analysisProb.append([0.035, 0.957, 0.004, 0.004])
  simulProb.append([0.035, 0.953, 0.005, 0.006])
  simulStd.append([0.0026, 0.0035, 0.0010, 0.0014])
  # 4
  analysisProb.append([0.017, 0.977, 0.003, 0.003])
  simulProb.append([0.012, 0.977, 0.004, 0.006])
  simulStd.append([0.0018, 0.0025, 0.0009, 0.0014])
  # 5
  analysisProb.append([0.088, 0.308, 0.088, 0.515])
  simulProb.append([0.039, 0.202, 0.032, 0.727])
  simulStd.append([0.0027, 0.0070, 0.0023, 0.0063])
  # 6
  analysisProb.append([0.205, 0.205, 0.021, 0.570])
  simulProb.append([0.144, 0.099, 0.019, 0.738])
  simulStd.append([0.0053, 0.0049, 0.0021, 0.0074])
  customKLD = 0.0342
  customKLDStd = 0.0016
  ########################################
  
  
plt.close("all")

imageSize = (1, 9)
imagePresentationDuration = 0.1 # used to be 20
dt = 0.001 # seconds

# 100/500 and 0.003 seems nice to me
firingRate = 42 # Hz;
AfiringRate = 0
numberXNeurons = imageSize[0] * imageSize[1] # 1 neurons per pixel (one for black) # input
numberYNeurons = 4 # output
numberZNeurons = 4 # prior

sigma = 0.01 # time frame in which spikes count as before output spike
c = 4 # 10 seems good, scales weights to 0 ... 1
tauRise = 0.001
tauDecay = 0.015  # might be onto something here, my problem gets better
# tauDecay = 0.015

learningRateFactor = 3
learningRate = 10**-learningRateFactor
RStar = 200 # Hz; total output firing rate

# initialize weights (for now between 0 and 1, not sure)
if loadWeights:
  allWeights = np.load("weights_98_440_4_c4.npy")
  weights = allWeights[:, 0::2]
  weightsInverted = allWeights[:, 1::2]
  priorWeights = np.load("priorWeights_98_440_4_c4.npy")
else:
  weights = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]],
                        "float64")
  weightsInverted = 1 - weights
  weights = np.log(weights)
  weightsInverted = np.log(weightsInverted)
  priorWeights = np.array([[0.9, 0.0333, 0.0333, 0.0333],
                           [0.0333, 0.9, 0.0333, 0.0333],
                           [0.0333, 0.0333, 0.9, 0.0333],
                           [0.0333, 0.0333, 0.0333, 0.9]],
                        "float64")
  priorWeights = np.log(priorWeights)

# generate Input data
# no random generation needed for this experiment, rather I use handpicked examples.
# image, prior = dataGenerator.generateRandom1DLineImage(imageSize)
images = [[],[],[]]
# 1
image = np.array([[255, 255, 0, 0, 0, 255, 255, 255, 255]], dtype=np.uint8)
prior = 1
images[0].append(image)
images[1].append(prior)
# for each pixel there are 2 neurons, this vector represents the active neurons for non active pixels
images[2].append(255 - image)
# 2
image = np.array([[255, 255, 0, 0, 0, 0, 255, 255, 255]], dtype=np.uint8)
prior = 1
images[0].append(image)
images[1].append(prior)
images[2].append(255 - image)
# 3
image = np.array([[255, 0, 0, 0, 255, 255, 255, 255, 255]], dtype=np.uint8)
prior = 1
images[0].append(image)
images[1].append(prior)
images[2].append(255 - image)
# 4
image = np.array([[255, 255, 0, 0, 255, 255, 255, 255, 255]], dtype=np.uint8)
prior = 1
images[0].append(image)
images[1].append(prior)
images[2].append(255 - image)

# 5
image = np.array([[255, 255, 0, 0, 0, 255, 255, 255, 255]], dtype=np.uint8)
prior = 3
images[0].append(image)
images[1].append(prior)
# for each pixel there are 2 neurons, this vector represents the active neurons for non active pixels
images[2].append(255 - image)
# 6
image = np.array([[255, 0, 0, 0, 255, 255, 255, 255, 255]], dtype=np.uint8)
prior = 3
images[0].append(image)
images[1].append(prior)
images[2].append(255 - image)

# TODO grid search
firingRateList = [90, 92, 94, 96, 98, 100, 110, 120]
AfiringRateList = [0]
# tauDecayList = [0.003, 0.005, 0.007, 0.015]

for gridIterator in range(1):
  # change parameters due to grid search
  #firingRate = firingRateList[gridIterator]
  # AfiringRate = AfiringRateList[gridIterator]
  
  PvonYvorausgesetztXundZSimulationListList = []
  for standardDeviationIterator in range(numberOfRuns):
  
    imagesEncoded = []
    priorsEncoded = []
    PvonYvorausgesetztXundZSimulationList = []
    # 1 simulation per handpicked input data
    for inputIterator in range(len(images[0])):
      XSpikes = [[],[]] # input
      XSpikesInverted = [[],[]] # input
      YSpikes = [[],[]] # output
      ZSpikes = [[],[]] # prior
      
      indexOfLastYSpike = [0] * numberXNeurons
      ZNeuronsRecievedYSpikes = [[],[]]
      
      # Metric to measure training progress
      # check how many different Z neurons fired during one image
      distinctZFiredHistory = []
      distinctZFired = []
      averageZFired = []
      averageZFiredHistory = []
      
      # pick current image and prior
      image = images[0][inputIterator]
      imageInverted = images[2][inputIterator]
      prior = images[1][inputIterator]
      
      # start simulation
      for t in np.arange(0, imagePresentationDuration, dt):
        # generate training data every 50ms
        if abs(t - round(t / imagePresentationDuration) * imagePresentationDuration) < 1e-10:
          distinctZFiredHistory.append(len(distinctZFired))
          distinctZFired = []
          if averageZFired:
            mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
            amountMostSpikingZ = averageZFired.count(mostSpikingZ)
            averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
            averageZFired = []
        
        # generate X Spikes for this step
        for i in range(image.shape[1]):
          # check if the Xi is active
          if image[0][i] == 0:
           # check if Xi spiked in this timestep
           if poissonGenerator.doesNeuronFire(firingRate, dt):
              # when did it spike
              XSpikes[0].append(t)
              # which X spiked
              XSpikes[1].append(i)
              
        # generate X SpikesInverted for this step
        for i in range(imageInverted.shape[1]):
          # check if the Xi is active
          if imageInverted[0][i] == 0:
           # check if Xi spiked in this timestep
           if poissonGenerator.doesNeuronFire(firingRate, dt):
              # when did it spike
              XSpikesInverted[0].append(t)
              # which X spiked
              XSpikesInverted[1].append(i)
             
        priorChoices = [0, 1, 2, 3]
        # generate A Spikes for this step
        if poissonGenerator.doesNeuronFire(AfiringRate, dt):
          ZSpikes[0].append(t)
          ZSpikes[1].append(prior)
    
      
        
        # Next we have to calculate Uk
        U = np.zeros(numberYNeurons)
        # Add contribution of X
        expiredYSpikeIDs = []
        XTilde = np.zeros(numberXNeurons)
        for i, XNeuron in enumerate(XSpikes[1]):
          # First mark all XSpikes older than sigma and do not use for calculation of Uk
          if XSpikes[0][i] < t - sigma:
            expiredYSpikeIDs.append(i)
          else:
            XTilde[XSpikes[1][i]] = kernel.tilde(t, dt, XSpikes[0][i], tauRise, tauDecay)
            for k in range(numberYNeurons):
              U[k] += weights[k, XNeuron] * XTilde[XSpikes[1][i]]
        # delete all spikes that are longer ago than sigma (10ms?) from XSpikes
        for toDeleteID in sorted(expiredYSpikeIDs, reverse=True):
          del XSpikes[0][toDeleteID]
          del XSpikes[1][toDeleteID]
          
        # Add contribution of XInverted
        expiredYSpikeIDs = []
        XTilde = np.zeros(numberXNeurons)
        for i, XNeuron in enumerate(XSpikesInverted[1]):
          # First mark all XSpikes older than sigma and do not use for calculation of Uk
          if XSpikesInverted[0][i] < t - sigma:
            expiredYSpikeIDs.append(i)
          else:
            XTilde[XSpikesInverted[1][i]] = kernel.tilde(t, dt, XSpikesInverted[0][i], tauRise, tauDecay)
            for k in range(numberYNeurons):
              U[k] += weightsInverted[k, XNeuron] * XTilde[XSpikesInverted[1][i]]
        # delete all spikes that are longer ago than sigma (10ms?) from XSpikes
        for toDeleteID in sorted(expiredYSpikeIDs, reverse=True):
          del XSpikesInverted[0][toDeleteID]
          del XSpikesInverted[1][toDeleteID]
          
        # Add contribution of A
        ATilde = np.zeros(numberZNeurons)
        expiredASpikeIDs = []
        for i in range(len(ZSpikes[1])):
          # First mark all ZSpikes older than sigma and do not use for calculation of Uk
          if ZSpikes[0][i] < t - sigma:
            expiredASpikeIDs.append(i)
          else:
            ATilde[ZSpikes[1][i]] = kernel.tilde(t, dt, ZSpikes[0][i], tauRise, tauDecay)
            for k in range(numberYNeurons):
              U[k] += priorWeights[prior, k] * ATilde[ZSpikes[1][i]]
        # delete all spikes that are longer ago than sigma (10ms?) from ZSpikes
        for toDeleteID in sorted(expiredASpikeIDs, reverse=True):
          del ZSpikes[0][toDeleteID]
          del ZSpikes[1][toDeleteID] 
      
        # calculate current Inhibition signal
        inhTMP = 0
        for i in range(numberYNeurons):
          inhTMP += np.exp(U[i])
        Iinh = - np.log(RStar) + np.log(inhTMP)
          
        # calc instantaneous fire rate for each Z Neuron for this time step
        r = np.zeros(numberYNeurons)
        ZNeuronsThatWantToFire = []
        ZNeuronWantsToFireAtTime = []
        # ZNeuronFireFactors is used to choose between multiple Z firing in this timestep
        ZNeuronFireFactors = []
        for k in range(numberYNeurons):
          r[k] = np.exp(U[k] - Iinh)
          # as far as i understand rk (titled "instantaneous fire rate" in nessler) just says
          # how many events occur per second on average
          ZkFires, ZNeuronFireFactor = poissonGenerator.doesZNeuronFire(r[k], dt) 
          if ZkFires:
            # mark that Zk wants to fire and also save the time it wants to fire at
            ZNeuronsThatWantToFire.append(k)
            ZNeuronWantsToFireAtTime.append(t)
            ZNeuronFireFactors.append(ZNeuronFireFactor)
        
        # TEST
        # let all output neurons fire that want to fire
        for i in range(len(ZNeuronsThatWantToFire)):
          YSpikes[0].append(ZNeuronsThatWantToFire[i])
          YSpikes[1].append(t)
          averageZFired.append(ZNeuronsThatWantToFire[i])
        # ORIGINAL
        # this was BULLSHIT BY ME. Nessler clearly stated that multiple Z may fire at the same time
        # This partially caused adjacent regions to be expressed weaker than they shouldve
        # # check if any Z Neurons want to fire and determine winner Z
        # if len(ZNeuronsThatWantToFire) > 0:
        #   ZFireFactorMax = -math.inf
        #   ZNeuronWinner = math.inf
        #   for i in range(len(ZNeuronsThatWantToFire)):
        #     if ZNeuronFireFactors[i] > ZFireFactorMax:  
        #       ZNeuronWinner = ZNeuronsThatWantToFire[i]
        #       ZFireFactorMax = ZNeuronFireFactors[i]      
        #   YSpikes[0].append(ZNeuronWinner)
        #   YSpikes[1].append(t)
        
          
        # NOTNEEDED
        #   averageZFired.append(ZNeuronWinner)
        #   # append ID of Z if this Z has not fired yet in this imagePresentationDuration
        #   if not distinctZFired.count(ZNeuronWinner):
        #     distinctZFired.append(ZNeuronWinner)
        #   # update weights of all Y to ZThatFired
        #   # do not update weights in this experiment for now we want to analyze the mathematically determined weights
        #   # weights = neuronFunctions.updateWeights(XTilde, weights, ZNeuronWinner, c, learningRate)
        #   # priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronWinner, c, learningRate)
          
      # Simulation DONE
      
      if useCustomValues:
        if customDoubleSize:
          directoryPath =  "customDouble_fInput" + str(customFInput) + "_fPrior" + str(customFPrior) + "_tauDecay" + str(customTauDecay)
        elif customC:
          directoryPath =  "custom_fInput" + str(customFInput) + "_fPrior" + str(customFPrior) + "_tauDecay" + str(customTauDecay) + "_c" + str(customC)
        else:  
          directoryPath =  "custom_fInput" + str(customFInput) + "_fPrior" + str(customFPrior) + "_tauDecay" + str(customTauDecay)
      else:
        directoryPath =  "Simulation_fInput" + str(firingRate) + "_fPrior" + str(AfiringRate) + "_tauDecay" + str(tauDecay) + "_c" + str(c)
      if not os.path.exists(directoryPath):
        os.mkdir(directoryPath)
      # np.save(directoryPath + "/weights" + ".npy", weights)
      # np.save(directoryPath + "/priorWeights" + ".npy", priorWeights)
        
      
      # 1 hot encode prior
      priorEncoded = np.zeros(4)
      # TODO  !!!!!!!! disabled next line to remove prior 
      priorEncoded[prior] = 1
      priorsEncoded.append(priorEncoded)
      # 1 hot encode image
      # were image has value 0 => black pixel => should become 1 in encoded image
      imageEncoded = np.zeros(9)
      for i in range(image.shape[1]):
        if image[0, i] == 0:
          imageEncoded[i] = 1
      imagesEncoded.append(imageEncoded)
      
      PvonYvorausgesetztXundZSimulation = np.zeros(4)
      totalSpikes = len(YSpikes[0])
      amountY0Spikes = len(np.where(np.array(YSpikes[0]) == 0)[0])
      amountY1Spikes = len(np.where(np.array(YSpikes[0]) == 1)[0])
      amountY2Spikes = len(np.where(np.array(YSpikes[0]) == 2)[0])
      amountY3Spikes = len(np.where(np.array(YSpikes[0]) == 3)[0])
      PvonYvorausgesetztXundZSimulation[0] = amountY0Spikes / totalSpikes
      PvonYvorausgesetztXundZSimulation[1] = amountY1Spikes / totalSpikes
      PvonYvorausgesetztXundZSimulation[2] = amountY2Spikes / totalSpikes
      PvonYvorausgesetztXundZSimulation[3] = amountY3Spikes / totalSpikes
      PvonYvorausgesetztXundZSimulationList.append(PvonYvorausgesetztXundZSimulation)
    PvonYvorausgesetztXundZSimulationListList.append(PvonYvorausgesetztXundZSimulationList)
  fig = plt.figure(figsize=(10, 12))
  # gs = fig.add_gridspec(2 * int(len(imagesEncoded)/2) + 1, 20)
  rowsMultiplicator = 16
  barPlotVerticalOffset = 8
  barPlotLeftHorizontalOffset = 1
  barPlotRightHorizontalOffset = 1
  inputDataOffset = 4
  analyisisMultiplicator = 2
  gs = fig.add_gridspec(rowsMultiplicator * int(len(imagesEncoded)/2) + 1, 20)
  counter = 0
  klDivergenceList = [[],[],[],[],[],[]]
  
  if customDoubleSize:
    images = [[],[],[]]
    # 1
    image = np.array([[255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 1
    images[0].append(image)
    images[1].append(prior)
    # for each pixel there are 2 neurons, this vector represents the active neurons for non active pixels
    images[2].append(255 - image)
    # 2
    image = np.array([[255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 1
    images[0].append(image)
    images[1].append(prior)
    images[2].append(255 - image)
    # 3
    image = np.array([[255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 1
    images[0].append(image)
    images[1].append(prior)
    images[2].append(255 - image)
    # 4
    image = np.array([[255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 1
    images[0].append(image)
    images[1].append(prior)
    images[2].append(255 - image)

    # 5
    image = np.array([[255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 3
    images[0].append(image)
    images[1].append(prior)
    # for each pixel there are 2 neurons, this vector represents the active neurons for non active pixels
    images[2].append(255 - image)
    # 6
    image = np.array([[255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)
    prior = 3
    images[0].append(image)
    images[1].append(prior)
    images[2].append(255 - image)
  
  for i in range(int(len(imagesEncoded)/2)):
    for j in range(2):
      imageEncoded = imagesEncoded[counter]
      image = images[0][counter]
      prior = images[1][counter]
      priorEncoded = priorsEncoded[counter]
      
      # calc std and average of the 20 runs
      PvonYvorausgesetztXundZSimulationMeanTmp = np.zeros(4)
      PvonYvorausgesetztXundZSimulationStdTmp = [[],[],[],[]]
      for runsIterator in range(numberOfRuns):
        for outputClassIterator in range(numberYNeurons):
          PvonYvorausgesetztXundZSimulationMeanTmp[outputClassIterator] += PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator]
          PvonYvorausgesetztXundZSimulationStdTmp[outputClassIterator].append(PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator])
      
      # calc mean of simulation probabs over 20 runs
      PvonYvorausgesetztXundZSimulationMeanTmp /= numberOfRuns
      PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulationMeanTmp
      
      # calc standard deviation of simulation probabs over 20 runs
      standardDeviations = np.zeros(numberYNeurons)
      for outputClassIterator in range(numberYNeurons):
        standardDeviations[outputClassIterator] = np.std(PvonYvorausgesetztXundZSimulationStdTmp[outputClassIterator])
        
      
      PvonYvorausgesetztXundZAnalysis = mathematischeAnalyse.calcPvonYvorausgesetztXundZ(imageEncoded, priorEncoded)
        
      # TODO changed function to disable prior
      # PvonYvorausgesetztXundZAnalysis = mathematischeAnalyse.calcPvonYvorausgesetztXundZNull(imageEncoded, priorEncoded)
      
      # gs2 = fig.add_gridspec(6, 2, wspace=0.4, hspace=25)
      # hspace seems to be capped and doesnt really influence the spacing anymore. 
      # but the .svg seems fine anyway, solve only if figure is not good enough.
      # gs3 = fig.add_gridspec(6, 2, wspace=0.4, hspace=400)
      
      if customDoubleSize:
        ax10 = fig.add_subplot(gs[(0 + rowsMultiplicator*i + inputDataOffset) : (0 + rowsMultiplicator*i + inputDataOffset) + analyisisMultiplicator, 0 +1 + 10*j:10 + 10*j - 1])
      else:
        ax10 = fig.add_subplot(gs[(0 + rowsMultiplicator*i + inputDataOffset) : (0 + rowsMultiplicator*i + inputDataOffset) + analyisisMultiplicator, 0 + 2 + 10*j:10 + 10*j - 2])
      
      # ax21 = fig.add_subplot(gs[1 + 2*i, 0 + 10*j:4 + 10*j])
      # ax22 = fig.add_subplot(gs[1 + 2*i, 6 + 10*j:10 + 10*j])
      ax20 = fig.add_subplot(gs[1 + rowsMultiplicator*i + barPlotVerticalOffset: rowsMultiplicator - 1 + rowsMultiplicator*i, barPlotLeftHorizontalOffset + 10*j:10 - barPlotRightHorizontalOffset + 10*j])
      
      # Add ghost axes and titles
      ax_firstRow = fig.add_subplot(gs[(0 + rowsMultiplicator*i + inputDataOffset) : (0 + rowsMultiplicator*i + inputDataOffset) + analyisisMultiplicator, 0 + 10*j:10 + 10*j])
      ax_firstRow.axis('off')
      ax_firstRow.set_title('A' + str(counter+1), loc="left", x=0.2,y=1, fontsize=16.0, fontweight='semibold')
      
      ax_secondRow = fig.add_subplot(gs[1 + rowsMultiplicator*i + barPlotVerticalOffset: rowsMultiplicator - 1 + rowsMultiplicator*i, 0 + 10*j])
      ax_secondRow.axis('off')
      ax_secondRow.set_title('B' + str(counter+1), loc="left", x=1.2,y=1, fontsize=16.0, fontweight='semibold')
      
      # ax_thirdRow = fig.add_subplot(gs[1 + 2*i, 6 + 10*j])
      # ax_thirdRow.axis('off')
      # ax_thirdRow.set_title('C' + str(counter+1), loc="left", x=-0.11,y=0.5, fontsize=16.0, fontweight='semibold')
      
      # plot input data
      ax10.imshow(images[0][counter], cmap='gray')
      if customEnablePrior:
        if customDoubleSize:
          rect = patches.Rectangle((-0.5 + prior*4,-0.5), 6, 1, linewidth=6, edgecolor='r', facecolor='none')
        else:
          rect = patches.Rectangle((-0.5 + prior*2,-0.5), 3, 1, linewidth=8, edgecolor='r', facecolor='none')
        ax10.add_patch(rect)
        rect.set_clip_path(rect)
      ax10.axvline(x=0.5)
      ax10.axvline(x=1.5)
      ax10.axvline(x=2.5)
      ax10.axvline(x=3.5)
      ax10.axvline(x=4.5)
      ax10.axvline(x=5.5)
      ax10.axvline(x=6.5)
      ax10.axvline(x=7.5)
      ax10.axvline(x=8.5)    
      if customDoubleSize:
        ax10.axvline(x=9.5)
        ax10.axvline(x=10.5)
        ax10.axvline(x=11.5)
        ax10.axvline(x=12.5)
        ax10.axvline(x=13.5)
        ax10.axvline(x=14.5)
        ax10.axvline(x=15.5)
        ax10.axvline(x=16.5)
        ax10.axvline(x=17.5)
      ax10.set_ylim([-0.5, 0.5])
      ax10.set_title("Input data", fontsize=14)
      ax10.axes.yaxis.set_visible(False)
      ax10.tick_params(axis='x', which='major', labelsize=12)
      if customDoubleSize:
        ax10.set_xticks([0,2,4,6,8,10, 12, 14, 16])
      else:
        ax10.set_xticks([0,1,2,3,4,5,6,7,8])

      
      x = np.arange(4)
      width = 0.4
      offset = width / 2
      ax20.bar(x - offset, analysisProb[counter], width=width, color='C3')
      ax20.bar(x + offset, simulProb[counter], yerr=simulStd[counter], width=width, color='C0')
      ax20.set_title('Output probabilities', fontsize=14)
      ax20.set_xticks(x, ["$y_1$", "$y_2$", "$y_3$", "$y_4$"])
      ax20.set_yticks(barYTicks)
      ax20.set_ylim(barYLim)
      ax20.tick_params(axis='x', which='major', labelsize=12)
      ax20.tick_params(axis='y', which='major', labelsize=12)
      # legend1 = patches.Patch(color='C3', label='Analysis')
      # legend2 = patches.Patch(color='C0', label='Simulation')
      # ax20.legend(handles=[legend1, legend2], loc='upper right', prop={'size': 10})


      
      # TODO commented out tables, Need them as separate table.
      # PvonYvorausgesetztXundZAnalysis = PvonYvorausgesetztXundZAnalysis.reshape(4,1)
      # tab21 = ax21.table(cellText=np.around(PvonYvorausgesetztXundZAnalysis, 3), bbox=[0.2, -0.2, 0.3, 1])
      # tab21.set_fontsize(14)
      # tab21.auto_set_column_width(0)
      # # tab21.scale(1,1.2)
      # ax21.axis('off')
      # ax21.set_title("Analysis output probabilities", y=0.9, fontsize=14)
      
      # PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulation.reshape(4,1)
      # standardDeviations = standardDeviations.reshape(4,1)
      # cellTextTmp = [[], [], [], []]
      # for cellTextIterator in range(numberYNeurons):  
      #   cellTextTmp[cellTextIterator].append(str(np.around(PvonYvorausgesetztXundZSimulation[cellTextIterator], 3)).strip("[]") + " (" + str(np.around(standardDeviations[cellTextIterator], 4)).strip("[]") + ")")
      # tab22 = ax22.table(cellText=cellTextTmp, bbox=[0.2, -0.2, 0.7, 1])
      # tab22.set_fontsize(14)
      # tab22.auto_set_column_width(0)
      # # tab22.scale(1,1.2)
      # ax22.axis('off')
      # ax22.set_title("Simulation output probabilities", y=0.9, fontsize=14)
      
      # calculate Kullback Leibler Divergence for each image and each run
      for runsIterator in range(numberOfRuns):
        klDivergenceTmp = 0
        for outputClassIterator in range(numberYNeurons):
          klDivergenceTmp += PvonYvorausgesetztXundZAnalysis[outputClassIterator] * np.log(PvonYvorausgesetztXundZAnalysis[outputClassIterator] / PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator]) 
        klDivergenceList[counter].append(klDivergenceTmp)
      
      counter += 1
  
  klDivergenceMeanPerRun = []
  for runIterator in range(numberOfRuns):
    klDivergenceMeanPerRunTmp = 0
    for imageIterator in range(6):
      klDivergenceMeanPerRunTmp += klDivergenceList[imageIterator][runIterator]
    klDivergenceMeanPerRun.append(klDivergenceMeanPerRunTmp / 6)
     
      
  klDivergenceMean = sum(klDivergenceMeanPerRun) / len(klDivergenceMeanPerRun)
  klDivergenceStd = np.std(klDivergenceMeanPerRun)  
  
  ax3 = fig.add_subplot(gs[rowsMultiplicator * int(len(imagesEncoded)/2):rowsMultiplicator * int(len(imagesEncoded)/2)+1, 0:20])
  ax3.axis('off')
  textStyle = dict(horizontalalignment='center', verticalalignment='center',
                  fontsize=16)
  if useCustomValues:
    ax3.text(0.5, -1.5, "Kullback–Leibler divergence = " + str(customKLD) + " $\pm$ " + str(customKLDStd), textStyle, transform=ax3.transAxes)
  else:
    ax3.text(0.5, -1.5, "Kullback–Leibler divergence = " + str(np.around(klDivergenceMean, 4)).strip("[]") + " $\pm$ " + str(np.around(klDivergenceStd, 5)), textStyle, transform=ax3.transAxes)
  
  legend1 = patches.Patch(color='C3', label='Analysis output probabilities')
  legend2 = patches.Patch(color='C0', label='Simulation output probabilities')
  ax3.legend(handles=[legend1, legend2], loc=(0.293, 56.5), prop={'size': 12})
  
  if useCustomValues:
    if customDoubleSize:
      pickle.dump(fig, open(directoryPath + "/doubleSize_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + '.pickle','wb'))
      plt.savefig(directoryPath + "/doubleSize_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + ".svg", bbox_inches='tight')  
      plt.savefig(directoryPath + "/doubleSize_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + ".png", bbox_inches='tight', dpi=300)
    elif customC:
      pickle.dump(fig, open(directoryPath + "/trainingEvaluation_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + "_c" + str(customC) + '.pickle','wb'))
      plt.savefig(directoryPath + "/trainingEvaluation_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + "_c" + str(customC) + ".svg", bbox_inches='tight')  
      plt.savefig(directoryPath + "/trainingEvaluation_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + "_c" + str(customC) + ".png", bbox_inches='tight', dpi=300)
    else:
      pickle.dump(fig, open(directoryPath + "/1D_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + '.pickle','wb'))
      plt.savefig(directoryPath + "/1D_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + ".svg", bbox_inches='tight')  
      plt.savefig(directoryPath + "/1D_" + str(customFInput) + "_" + str(customFPrior) + "_" + str(customTauDecay) + ".png", bbox_inches='tight', dpi=300)
  else:
    pickle.dump(fig, open(directoryPath + "/trainingPlot5" + '.pickle','wb'))
    plt.savefig(directoryPath + "/trainingPlot5" + ".svg", bbox_inches='tight')  
    plt.savefig(directoryPath + "/trainingPlot5" + ".png", bbox_inches='tight', dpi=300)
  # plt.show()
  plt.close()
