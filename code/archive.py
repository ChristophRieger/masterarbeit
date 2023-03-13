# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:36:47 2023

@author: chris
"""

# plot YSpikes
for i in range(numberYNeurons): 
  tmpList = [x for x in YSpikes[i] if x < 0.2]
  plt.scatter(tmpList, [i] * len(tmpList), s=1, c='black')