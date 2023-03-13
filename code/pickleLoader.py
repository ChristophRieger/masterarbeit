# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:09:24 2023

@author: chris
"""

import pickle
import matplotlib.pyplot as plt


plt.figure()
plt.plot(pickle.load("ZSpikes.pickle"))
