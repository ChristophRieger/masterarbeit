# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:24:47 2023

@author: chris
"""

import dataGenerator
import matplotlib.pyplot as plt


fig, angle = dataGenerator.generateImage(7920)
plt.imshow(fig, cmap='gray')