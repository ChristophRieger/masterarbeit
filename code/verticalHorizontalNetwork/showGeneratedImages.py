# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:56:48 2023

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')

import dataGenerator
import matplotlib.pyplot as plt

plt.close("all")

# generate vertical images
image, xPosition, prior, orientation = dataGenerator.generateRandomVerticalLineImage()
plt.figure()
plt.imshow(image, cmap='gray')

# image, xPosition, prior, orientation = dataGenerator.generateVerticalLineImage(29)
# plt.figure()
# plt.imshow(image, cmap='gray')

# generate horizontal images
image, yPosition, prior, orientation = dataGenerator.generateRandomHorizontalLineImage()
plt.figure()
plt.imshow(image, cmap='gray')

# image, yPosition, prior, orientation = dataGenerator.generateHorizontalLineImage(0)
# plt.figure()
# plt.imshow(image, cmap='gray')

image, xPosition, yPosition, orientation = dataGenerator.generateRandomCrossLineImage()
plt.figure()
plt.imshow(image, cmap='gray')