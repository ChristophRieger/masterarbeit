# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 17:57:05 2023

@author: chris
"""

import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

ones_array = np.ones(shape=(8,))
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'white', 'white', 'white',]
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.figure()
# startangle 90 is at top
plt.pie(ones_array, colors=colors, startangle = 90, counterclock=False,)
plt.show() 