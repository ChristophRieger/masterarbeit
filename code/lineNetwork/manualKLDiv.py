# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:52:08 2023

@author: chris
"""
import numpy as np

##########################
# This script was used to calculate mean and std KLD
##########################


# single 440
klDivergenceListSingle440 = []
# 1
analysis = [0.01, 0.977, 0.01, 0.002]
sim = [0.009, 0.979, 0.009, 0.004]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# 2
analysis = [0.01, 0.967, 0.021, 0.002]
sim = [0.009, 0.963, 0.025, 0.003]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# 3
analysis = [0.035, 0.957, 0.004, 0.004]
sim = [0.067, 0.916, 0.007, 0.009]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# 4
analysis = [0.017, 0.977, 0.003, 0.003]
sim = [0.025, 0.955, 0.009, 0.011]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# 5
analysis = [0.088, 0.308, 0.088, 0.515]
sim = [0.051, 0.379, 0.046, 0.523]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# 6
analysis = [0.205, 0.205, 0.021, 0.57]
sim = [0.191, 0.188, 0.022, 0.6]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListSingle440.append(klDivergenceTmp)
# Mean
klDivergenceMeanSingle440 = sum(klDivergenceListSingle440) / len(klDivergenceListSingle440)

# double 440
klDivergenceListDouble440 = []
# 1
analysis = [0.01, 0.977, 0.01, 0.002]
sim = [0.002, 0.996, 0.002, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# 2
analysis = [0.01, 0.967, 0.021, 0.002]
sim = [0.002, 0.982, 0.016, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# 3
analysis = [0.035, 0.957, 0.004, 0.004]
sim = [0.089, 0.907, 0.001, 0.002]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# 4
analysis = [0.017, 0.977, 0.003, 0.003]
sim = [0.016, 0.979, 0.002, 0.003]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# 5
analysis = [0.088, 0.308, 0.088, 0.515]
sim = [0.022, 0.799, 0.019, 0.16]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# 6
analysis = [0.205, 0.205, 0.021, 0.57]
sim = [0.356, 0.352, 0.007, 0.285]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble440.append(klDivergenceTmp)
# Mean
klDivergenceMean440 = sum(klDivergenceListDouble440) / len(klDivergenceListDouble440)

# double 880
klDivergenceListDouble880 = []
# 1
analysis = [0.01, 0.977, 0.01, 0.002]
sim = [0.0000001, 1, 0.0000001, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# 2
analysis = [0.01, 0.967, 0.021, 0.002]
sim = [0.0000001, 1, 0.0000001, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# 3
analysis = [0.035, 0.957, 0.004, 0.004]
sim = [0.003, 0.997, 0.0000001, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# 4
analysis = [0.017, 0.977, 0.003, 0.003]
sim = [0.0000001, 1, 0.0000001, 0.0000001]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# 5
analysis = [0.088, 0.308, 0.088, 0.515]
sim = [0.007, 0.369, 0.005, 0.619]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# 6
analysis = [0.205, 0.205, 0.021, 0.57]
sim = [0.106, 0.109, 0.001, 0.783]
klDivergenceTmp = 0
for outputClassIterator in range(4):
  klDivergenceTmp += analysis[outputClassIterator] * np.log(analysis[outputClassIterator] / sim[outputClassIterator]) 
klDivergenceListDouble880.append(klDivergenceTmp)
# Mean
klDivergenceMean880 = sum(klDivergenceListDouble880) / len(klDivergenceListDouble880)