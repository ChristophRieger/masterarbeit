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

fig, (ax1, ax2) = plt.subplots(1,2)

fig.suptitle('Examples of training images', fontsize=14)
fig.subplots_adjust(top=1.1)
image1, xPosition, prior, orientation = dataGenerator.generateVerticalLineImage(12)
image2, yPosition, prior, orientation = dataGenerator.generateHorizontalLineImage(18)
# Line plots
ax1.set_title('Image in vertical orientation')
ax1.imshow(image1, cmap='gray')
ax2.set_title('Image in horizontal orientation')
ax2.imshow(image2, cmap='gray')



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

image, position, orientation = dataGenerator.generateRandomCrossLineImage()
plt.figure()
plt.imshow(image, cmap='gray')