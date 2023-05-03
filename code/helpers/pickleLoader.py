# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:09:24 2023

@author: chris
"""

from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt


plt.close("all")

current_directory = Path(__file__).parent #Get current directory
# file = open(os.path.join(current_directory, '../code/angleNetworkAdaptiveInhibition/', 'trainingPlot.pickle'), 'rb') #rb = read bytes because we are reading the file
file = pickle.load(open('C://Users/chris/Desktop/Masterarbeit/code/angleNetworkAdaptiveInhibition/trainingPlot.pickle', 'rb')) #rb = read bytes because we are reading the file)

file.show()
# plt.figure()
# plt.plot(pickle.load(file))
