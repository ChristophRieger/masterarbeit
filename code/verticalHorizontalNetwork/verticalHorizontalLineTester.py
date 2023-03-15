# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:56:48 2023

@author: chris
"""

import dataGenerator
import matplotlib.pyplot as plt


image, xPosition = dataGenerator.generateRandomVerticalLineImage()
plt.figure()
plt.imshow(image, cmap='gray')

image, yPosition = dataGenerator.generateRandomHorizontalLineImage()
plt.figure()
plt.imshow(image, cmap='gray')