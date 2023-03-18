# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:04:22 2023

@author: chris
"""
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

plt.close("all")

# Command Center
plotWeights = False

imageSize = (29, 29)
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz
numberYNeurons = imageSize[0] * imageSize[1] * 2
numberZNeurons = 10
tauInh = 0.005
sigma = 0.01 
tauRise = 0.001
tauDecay = 0.015

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']
pieColors = []

weights = np.load("c20_3_YZWeights.npy")
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
  Iinh = 0
  IinhStartTime = 0
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
      # calculate inhibition signal and store time of last Z spike
      inhTMP = 0
      for i in range(numberZNeurons):
        inhTMP += np.exp(U[i])
      Iinh = np.log(inhTMP)
      IinhStartTime = t
    elif t - IinhStartTime > tauInh:
      Iinh = 0
      IinhStartTime = math.inf
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
  
fig_object = plt.figure()
# startangle 90 is at top
plt.pie(threesixtyAngles, colors=pieColors, startangle = 90, counterclock=False,)
plt.title("Most active Z neuron depending on angle")
pieLegend1 = patches.Patch(color=colors[0], label='Z1')
pieLegend2 = patches.Patch(color=colors[1], label='Z2')
pieLegend3 = patches.Patch(color=colors[2], label='Z3')
pieLegend4 = patches.Patch(color=colors[3], label='Z4')
pieLegend5 = patches.Patch(color=colors[4], label='Z5')
pieLegend6 = patches.Patch(color=colors[5], label='Z6')
pieLegend7 = patches.Patch(color=colors[6], label='Z7')
pieLegend8 = patches.Patch(color=colors[7], label='Z8')
pieLegend9 = patches.Patch(color=colors[8], label='Z9')
pieLegend10 = patches.Patch(color=colors[9], label='Z10')
plt.legend(handles=[pieLegend1,pieLegend2,pieLegend3,pieLegend4,pieLegend5,pieLegend6,pieLegend7,pieLegend8,pieLegend9,pieLegend10], loc=(1.04, 0.25))
pickle.dump(fig_object, open('pie.pickle','wb'))

# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
distinctZFiredHistory.pop(0)
fig_object = plt.figure()
plt.plot(distinctZFiredHistory)
plt.title("Number of distinct Z neurons spiking")
plt.ylabel("Number of distinct Z neurons spiking")
plt.xlabel("Image shown")
pickle.dump(fig_object, open('distinctZ.pickle','wb'))

fig_object = plt.figure()
for i in range(0, len(ZSpikeHistory[0])):
  plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
plt.title("Z Spikes")
plt.ylabel("Z Neuron")
plt.xlabel("t [s]")
pickle.dump(fig_object, open('ZSpikes.pickle','wb'))
      
# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
plt.figure()
plt.plot(averageZFiredHistory)
plt.title("Certainty of network")
plt.ylabel("Homogeneity of Z spikes")
plt.xlabel("Image shown")
pickle.dump(fig_object, open('averageZ.pickle','wb'))