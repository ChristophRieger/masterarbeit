# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:24:01 2024

@author: chris
"""
import numpy as np

analysisProb = []
simulProb = []
simulStd = []
# 1
analysisProb.append([0.010, 0.977, 0.010, 0.002])
simulProb.append([0.0000001, 1.000, 0.0000001, 0.0000001])
simulStd.append([0.0001, 0.0002, 0.0001, 0.0000])
# 2
analysisProb.append([0.010, 0.967, 0.021, 0.002])
simulProb.append([0.0000001, 1.000, 0.0000001, 0.0000001])
simulStd.append([0.0001, 0.0003, 0.0002, 0.0001])
# 3
analysisProb.append([0.035, 0.957, 0.004, 0.004])
simulProb.append([0.003, 0.997, 0.0000001, 0.0000001])
simulStd.append([0.0010, 0.0010, 0.0001, 0.0001])
# 4
analysisProb.append([0.017, 0.977, 0.003, 0.003])
simulProb.append([0.0000001, 1.000, 0.0000001, 0.0000001])
simulStd.append([0.0004, 0.0004, 0.0001, 0.0001])
# 5
analysisProb.append([0.088, 0.308, 0.088, 0.515])
simulProb.append([0.007, 0.369, 0.005, 0.619])
simulStd.append([0.0010, 0.0096, 0.0010, 0.0098])
# 6
analysisProb.append([0.205, 0.205, 0.021, 0.570])
simulProb.append([0.106, 0.109, 0.001, 0.783])
simulStd.append([0.0060, 0.0042, 0.0004, 0.0091])




klDivergenceList = []
for imageIterator in range(6):
  klDivergenceTmp = 0
  for outputClassIterator in range(4):
    klDivergenceTmp += analysisProb[imageIterator][outputClassIterator] * np.log(analysisProb[imageIterator][outputClassIterator] / simulProb[imageIterator][outputClassIterator]) 
  klDivergenceList.append(klDivergenceTmp)
  
klDivergence = np.mean(klDivergenceList)