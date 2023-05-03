# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:04:22 2023

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dataGenerator
import dataEncoder
import poissonGenerator
import kernel
import math
import pickle
import sys
import os

plt.close("all")

# Command Center
plotWeights = False

imageSize = (29, 29)
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz
numberYNeurons = imageSize[0] * imageSize[1] * 2
numberZNeurons = 10
sigma = 0.01 
tauRise = 0.001
tauDecay = 0.015
RStar = 200 # Hz; total output firing rate

c = 20
learningRateFactor = 3

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']
pieColors = []

weights = np.load("c" + str(c) + "_" + str(learningRateFactor) + "_YZWeights.npy")
intrinsicWeights = np.zeros(numberZNeurons)
# plot all weights
if plotWeights:
  for z in range(np.shape(weights)[1]):
    plt.figure()
    w = weights[0::2, z]
    w = w.reshape((29, 29))
    plt.imshow(w, cmap='gray')
  sys.exit()
    

# for each degree generate a line image and see which Z neuron spikes most and 
# how many distinct Z spike
distinctZFiredHistory = []
distinctZFired = []
averageZFired = []
averageZFiredHistory = []
ZSpikeHistory = [[],[]]

for angleIterator in range(0,180):
  YSpikes = [[],[]]
  ZSpikes = [[],[]]
  images = [[],[]]
  image, angle = dataGenerator.generateImage(angleIterator)
  images[0].append(image)
  images[1].append(angle)
  encodedImage = dataEncoder.encodeImage(image)
  distinctZFiredHistory.append(len(distinctZFired))
  distinctZFired = []

  if averageZFired:
    mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
    amountMostSpikingZ = averageZFired.count(mostSpikingZ)
    averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
    averageZFired = []
  
  for t in np.arange(0, imagePresentationDuration, dt):
    # generate Y Spikes for this step
    for i in range(len(encodedImage)):
      # check if the Yi is active
      if encodedImage[i] == 1:
       # check if Yi spiked in this timestep
       if poissonGenerator.doesNeuronFire(firingRate, dt):
         # when did it spike
         YSpikes[0].append(t)
         # which Y spiked
         YSpikes[1].append(i)
    U = np.zeros(numberZNeurons) + intrinsicWeights
  
    # Next we have to calculate Uk
    expiredYSpikeIDs = []
    YTilde = np.zeros(numberYNeurons)
    for i, YNeuron in enumerate(YSpikes[1]):
      # First mark all YSpikes older than sigma and do not use for calculation of Uk
      if YSpikes[0][i] < t - sigma:
        expiredYSpikeIDs.append(i)
      else:
        YTilde[YSpikes[1][i]] = kernel.tilde(t, dt, YSpikes[0][i], tauRise, tauDecay)
        for k in range(numberZNeurons):
          U[k] += weights[YNeuron, k] * YTilde[YSpikes[1][i]]
    # delete all spikes that are longer ago than sigma (10ms?) from YSpikes
    for toDeleteID in sorted(expiredYSpikeIDs, reverse=True):
      del YSpikes[0][toDeleteID]
      del YSpikes[1][toDeleteID]
      
    # calculate current Inhibition signal
    inhTMP = 0
    for i in range(numberZNeurons):
      inhTMP += np.exp(U[i])
    Iinh = - np.log(RStar) + np.log(inhTMP)
  
    # calc instantaneous fire rate for each Z Neuron for this time step
    r = np.zeros(numberZNeurons)
    ZNeuronsThatWantToFire = []
    ZNeuronWantsToFireAtTime = []
    # ZNeuronFireFactors is used to choose between multiple Z firing in this timestep
    ZNeuronFireFactors = []
    for k in range(numberZNeurons):
      r[k] = np.exp(U[k] - Iinh)
      # as far as i understand rk (titled "instantaneous fire rate" in nessler) just says
      # how many events occur per second on average
      ZkFires, ZNeuronFireFactor = poissonGenerator.doesZNeuronFire(r[k], dt) 
      if ZkFires:
        # mark that Zk wants to fire and also save the time it wants to fire at
        ZNeuronsThatWantToFire.append(k)
        ZNeuronWantsToFireAtTime.append(t)
        ZNeuronFireFactors.append(ZNeuronFireFactor)
      
    # check if any Z Neurons want to fire and determine winner Z
    if len(ZNeuronsThatWantToFire) > 0:
      ZFireFactorMax = -math.inf
      ZNeuronWinner = math.inf
      for i in range(len(ZNeuronsThatWantToFire)):
        if ZNeuronFireFactors[i] > ZFireFactorMax:  
          ZNeuronWinner = ZNeuronsThatWantToFire[i]
          ZFireFactorMax = ZNeuronFireFactors[i]      
      ZSpikes[0].append(ZNeuronWinner)
      ZSpikeHistory[0].append(ZNeuronWinner)
      ZSpikes[1].append(t)
      averageZFired.append(ZNeuronWinner)
      ZSpikeHistory[1].append(t + angleIterator * imagePresentationDuration)
      # append ID of Z if this Z has not fired yet in this imagePresentationDuration
      if not distinctZFired.count(ZNeuronWinner):
        distinctZFired.append(ZNeuronWinner)

  # determine distinctZ
  
  # calc which Z fired the most for this angle
  whichZFired = np.zeros(numberZNeurons)
  for i in range(len(images[0])):
    # check which degree the image was
    currentAngle = angleIterator
    # iterate over all zSpikes and increment the Z that fired
    for j in range(len(ZSpikes[0])):
      whichZFired[ZSpikes[0][j]] += 1
  winnerID = math.inf
  maxSpikes = 0
  for j in range(numberZNeurons):
    if whichZFired[j] > maxSpikes:
      winnerID = j
      maxSpikes = whichZFired[j]
  # code winnerID of Z neuron to its color
  if winnerID == math.inf:
    pieColors.append('white')
  else:
    pieColors.append(colors[winnerID])
  
  print("Finished simulation of angle " + str(angleIterator) + "°")
  
# plot piechart of most fired Z per degree
threesixtyAngles = np.ones(360)
for i in range(180, 360):
  pieColors.append('white')
  
  
directoryPath =  "validation_c" + str(c) + "_eta" + str(learningRateFactor)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)
  
fig = plt.figure()
gs = fig.add_gridspec(2, 2, wspace=1, hspace=1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# startangle 90 is at top
ax1.pie(threesixtyAngles, colors=pieColors, startangle = 90, counterclock=False,)
ax1.set_title("Most active output neuron")
pieLegend1 = patches.Patch(color=colors[0], label='y1')
pieLegend2 = patches.Patch(color=colors[1], label='y2')
pieLegend3 = patches.Patch(color=colors[2], label='y3')
pieLegend4 = patches.Patch(color=colors[3], label='y4')
pieLegend5 = patches.Patch(color=colors[4], label='y5')
pieLegend6 = patches.Patch(color=colors[5], label='y6')
pieLegend7 = patches.Patch(color=colors[6], label='y7')
pieLegend8 = patches.Patch(color=colors[7], label='y8')
pieLegend9 = patches.Patch(color=colors[8], label='y9')
pieLegend10 = patches.Patch(color=colors[9], label='y10')
ax1.legend(handles=[pieLegend1,pieLegend2,pieLegend3,pieLegend4,pieLegend5,pieLegend6,pieLegend7,pieLegend8,pieLegend9,pieLegend10], loc=(1.04, 0.25))

# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
distinctZFiredHistory.pop(0)
ax3.plot(distinctZFiredHistory)
ax3.set_title("Distinct output neurons spiking")
ax3.set_ylabel("Number of distinct output neurons spiking")
ax3.set_xlabel("Image shown")

for i in range(0, len(ZSpikeHistory[0])):
  ax2.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
ax2.set_title("Output spikes")
ax2.set_ylabel("Output neuron")
ax2.set_xlabel("t [s]")
      
# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
ax4.plot(averageZFiredHistory)
ax4.set_title("Certainty of network")
ax4.set_ylabel("Homogeneity of output spikes")
ax4.set_xlabel("Image shown")

pickle.dump(fig, open(directoryPath + '/validation.pickle','wb'))
plt.savefig(directoryPath + "/validation.svg") 
plt.show()
