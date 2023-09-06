# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:35:09 2023

@author: chris
"""

import matplotlib.pyplot as plt

plt.close("all")

# Without Prior
# tauDecay = 0.015
fInputWithout15 =          [10     , 20     , 30     , 34     , 36     , 38     , 40     , 42     , 44     , 46     , 48     , 50     , 60     , 70     , 80     , 90     , 100    , 110]
KLDivergenceWithout15 =    [0.1916 , 0.0935 , 0.0424 , 0.0319 , 0.0274 , 0.0255 , 0.0251 , 0.0243 , 0.0246 , 0.0255 , 0.0284 , 0.0305 , 0.0545 , 0.0914 , 0.1372 , 0.1898 , 0.2521 , 0.3145]                      
KLDivergenceStdWithout15 = [0.00130, 0.00293, 0.00230, 0.00174, 0.00148, 0.00132, 0.00159, 0.00156, 0.00114, 0.00182, 0.00147, 0.00235, 0.00304, 0.00341, 0.00443, 0.00472, 0.00796, 0.01138]                                                      

plt.figure()
plt.plot(fInputWithout15, KLDivergenceWithout15)

plt.figure()
plt.errorbar(fInputWithout15, KLDivergenceWithout15, KLDivergenceStdWithout15, marker='.', ecolor="Red",)


# tauDecay = 0.004
fInputWithout4 = []
KLDivergenceWithout4 = []

#  With Prior
# tauDecay = 0.015
fInputWith15 = []
KLDivergenceWith15 = []

# tauDecay = 0.004
fInputWith4 = []
KLDivergenceWith4 = []