# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 09:55:32 2022

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')

import dataEncoder
import dataGenerator
import poissonGenerator
import neuronFunctions
import kernel

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import os
import pickle


# Command Center
loadWeights = False
disableIntrinsicWeights = True

plt.close("all")


# !!! i abandoned this ideas, as the active and nonactive neurons alternate, thus making it very hard to see anything
# !!! maybe i could only plot all black neurons...
# encodedImage = dataEncoder.encodeImage(image1[0])
# encodedImageArray = np.array(encodedImage)
# encodedImageArray = np.transpose(encodedImageArray)
# widenedEncodedImageArray = np.zeros((encodedImageArray.size, 1000))
# for i in range(0, 1000):
#   widenedEncodedImageArray[:,i] = encodedImageArray
# plt.figure()
# plt.imshow(widenedEncodedImageArray[:10,:1000], cmap='gray')

# took 29, so there is an actual center, which makes everything symmetric (the mask primarily)
imageSize = (29, 29)
simulationTime = 800 # seconds
# legi suggested to increase this from 0.05 to 0.2, works better
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz; Input neurons yn should spike with 20Hz => firingRate (Lambda) = 20/second
RStar = 200 # Hz; total output firing rate
numberYNeurons = imageSize[0] * imageSize[1] * 2 # 2 neurons per pixel (one for black, one for white)
numberZNeurons = 10

sigma = 0.01 # time frame in which spikes count as before output spike
c = 20 # 10 seems good, scales weights to 0 ... 1
tauRise = 0.001
tauDecay = 0.015
learningRateFactor = 3
learningRate = 10**-learningRateFactor

YSpikes = [[],[]]
ZSpikes = [[],[]]

# initialize weights (for now between 0 and 1, not sure)
if loadWeights:
  weights = np.load("")
  intrinsicWeights = np.zeros(numberZNeurons)
else:
  weights = np.full((numberYNeurons, numberZNeurons), 2, "float64")
  intrinsicWeights = np.zeros(numberZNeurons)

indexOfLastYSpike = [0] * numberYNeurons
ZNeuronsRecievedYSpikes = [[],[]]
images = [[],[]]

# Metric to measure training progress
# check how many different Z neurons fired during one image
distinctZFiredHistory = []
distinctZFired = []
averageZFired = []
averageZFiredHistory = []

# start simulation
for t in np.arange(0, simulationTime, dt):
  # generate training data every 50ms
  if abs(t - round(t / imagePresentationDuration) * imagePresentationDuration) < 1e-10:
    image, angle = dataGenerator.generateRandomlyOrientedLineImage(imageSize)
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


  # I have to add wk0 only once, thats why its here
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
    ZSpikes[1].append(t)
    averageZFired.append(ZNeuronWinner)
    # append ID of Z if this Z has not fired yet in this imagePresentationDuration
    if not distinctZFired.count(ZNeuronWinner):
      distinctZFired.append(ZNeuronWinner)
    # update weights of all Y to ZThatFired
    weights = neuronFunctions.updateWeights(YTilde, weights, ZNeuronWinner, c, learningRate)
    if not disableIntrinsicWeights:
      intrinsicWeights = neuronFunctions.updateIntrinsicWeights(intrinsicWeights, ZNeuronWinner, c, learningRate)
    
  if (t) % 1 == 0:
    print("Finished simulation of t= " + str(t))
  
  
directoryPath =  "c" + str(c) + "_eta" + str(learningRateFactor)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)
np.save(directoryPath + "/c" + str(c) + "_eta" + str(learningRateFactor) + "_YZWeights.npy", weights)
  
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']

fig = plt.figure()
gs = fig.add_gridspec(4, 10)
gs2 = fig.add_gridspec(4, 2, wspace=0.4, hspace=3)

ax10 = fig.add_subplot(gs[0, 0])
ax11 = fig.add_subplot(gs[0, 1])
ax12 = fig.add_subplot(gs[0, 2])
ax13 = fig.add_subplot(gs[0, 3])
ax14 = fig.add_subplot(gs[0, 4])
ax15 = fig.add_subplot(gs[0, 5])
ax16 = fig.add_subplot(gs[0, 6])
ax17 = fig.add_subplot(gs[0, 7])
ax18 = fig.add_subplot(gs[0, 8])
ax19 = fig.add_subplot(gs[0, 9])

ax20 = fig.add_subplot(gs[1, 0])
ax21 = fig.add_subplot(gs[1, 1])
ax22 = fig.add_subplot(gs[1, 2])
ax23 = fig.add_subplot(gs[1, 3])
ax24 = fig.add_subplot(gs[1, 4])
ax25 = fig.add_subplot(gs[1, 5])
ax26 = fig.add_subplot(gs[1, 6])
ax27 = fig.add_subplot(gs[1, 7])
ax28 = fig.add_subplot(gs[1, 8])
ax29 = fig.add_subplot(gs[1, 9])

ax31 = fig.add_subplot(gs2[2:4, 0])
ax32 = fig.add_subplot(gs2[2:4, 1])

# Add ghost axes and titles
ax_firstRow = fig.add_subplot(gs[0, :])
ax_firstRow.axis('off')
ax_firstRow.set_title('A', loc="left", x=-0.04,y=0.5, fontsize=16.0, fontweight='semibold')

ax_secondRow = fig.add_subplot(gs[1, :])
ax_secondRow.axis('off')
ax_secondRow.set_title('B', loc="left", x=-0.04,y=0.5, fontsize=16.0, fontweight='semibold')

ax_thirdRowFirstColumn = fig.add_subplot(gs2[2:4, 0])
ax_thirdRowFirstColumn.axis('off')
ax_thirdRowFirstColumn.set_title('C', loc="left", x=-0.1, fontsize=16.0, fontweight='semibold')

ax_thirdRowSecondColumn = fig.add_subplot(gs2[2:4, 1])
ax_thirdRowSecondColumn.axis('off')
ax_thirdRowSecondColumn.set_title('D', loc="left", x=-0.1, fontsize=16.0, fontweight='semibold')

# plot training data examples and weights
for z in range(np.shape(weights)[1]):
  image = dataGenerator.generateImage(0 + 18*z)
  eval("ax1" + str(z) + ".imshow(image[0], cmap='gray')")
  eval("ax1" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")
  
  w = weights[0::2, z]
  w = w.reshape((29, 29))
  eval("ax2" + str(z) + ".imshow(w, cmap='gray')")
  eval("ax2" + str(z) + ".set_title('$w_{' + str(z+1) + 'n}$')")
  eval("ax2" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")


# plot the first 120 Z spikes
for i in range(0, 120):
  ax31.vlines(ZSpikes[1][i], ymin=ZSpikes[0][i] + 1 - 0.5, ymax=ZSpikes[0][i] + 1 + 0.5, color=colors[ZSpikes[0][i]], linewidth=0.5)
ax31.axvline(x=0, color="red", linestyle="dashed")
ax31.axvline(x=0.2, color="red", linestyle="dashed")
ax31.axvline(x=0.4, color="red", linestyle="dashed")
ax31.axvline(x=0.6, color="red", linestyle="dashed")
ax31.set_title("Output before learning")
ax31.set_ylabel("Output neuron")
ax31.set_xlabel("Time [s]")

# plot the last 120 Z spikes
for i in range(len(ZSpikes[0]) - len(ZSpikes[0][-120:]), len(ZSpikes[0])):
  ax32.vlines(ZSpikes[1][i], ymin=ZSpikes[0][i] + 1 - 0.5, ymax=ZSpikes[0][i] + 1 + 0.5, color=colors[ZSpikes[0][i]], linewidth=0.5)
ax32.axvline(x=simulationTime, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.2, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.4, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.6, color="red", linestyle="dashed")
ax32.set_title("Output after learning")
ax32.set_ylabel("Output neuron")
ax32.set_xlabel("Time [s]")

plt.show()
pickle.dump(fig, open(directoryPath + "/trainingPlot" + '.pickle','wb'))
plt.savefig(directoryPath + "/trainingPlot.svg")  


# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
distinctZFiredHistory.pop(0)
fig = plt.figure()
plt.plot(distinctZFiredHistory)
plt.title("Training progress")
plt.ylabel("Number of distinct output neurons spiking")
plt.xlabel("Image shown")
pickle.dump(fig, open(directoryPath + "/distinctY" + '.pickle','wb'))
plt.savefig(directoryPath + "/distinctY.svg") 

# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
fig = plt.figure()
plt.plot(averageZFiredHistory)
plt.title("Certainty of network")
plt.ylabel("Homogeneity of output spikes")
plt.xlabel("Image shown")
pickle.dump(fig, open(directoryPath + "/averageY" + '.pickle','wb'))
plt.savefig(directoryPath + "/averageY.svg") 

# output firing rate

outputFiringRate = []
for i in range(len(images[0])):
  ZSpikesForThisImage = 0
  for j in range(len(ZSpikes[0])):
    # get all spikes between t and t+imagePresentationDuration
    if ZSpikes[1][j] > i * imagePresentationDuration and ZSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
      ZSpikesForThisImage += 1
  outputFiringRate.append(ZSpikesForThisImage / imagePresentationDuration)

fig = plt.figure()
plt.plot(outputFiringRate)
plt.title("Output firing rate")
plt.ylabel("Firing rate [Hz]")
plt.xlabel("Image shown")
pickle.dump(fig, open(directoryPath + "/outputFiringRate" + '.pickle','wb'))
plt.savefig(directoryPath + "/outputFiringRate.svg") 












# # calc which Z fired the most for an angle
# angleAndWhichZFired = np.zeros([360, numberZNeurons])
# for i in range(len(images[0])):
#   # check which degree the image was
#   for angle in range(360):
#     if images[1][i] > angle and images[1][i] <= angle + 1:
#       currentAngle = angle
#   for j in range(len(ZSpikes[0])):
#     # get all spikes between t and t+imagePresentationDuration
#     # !!! maybe save last index to not always start from 0 like above for YNeuronspikes
#     if ZSpikes[1][j] > i * imagePresentationDuration and ZSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
#       angleAndWhichZFired[currentAngle, ZSpikes[0][j]] += 1

# # determine Z neuron that fired the most for each degree
# pieColors = []
# for i in range(360):
#   winnerID = math.inf
#   maxSpikes = 0
#   for j in range(numberZNeurons):
#     if angleAndWhichZFired[i][j] > maxSpikes:
#       winnerID = j
#       maxSpikes = angleAndWhichZFired[i][j]
#   # code winnerID of Z neuron to its color used above
#   if winnerID == math.inf:
#     pieColors.append('white')
#   else:
#     pieColors.append(colors[winnerID])

# # plot piechart of most fired Z per degree
# threesixtyAngles = np.ones(360)
# # startangle 90 is at top
# ax31.pie(threesixtyAngles, colors=pieColors, startangle = 90, counterclock=False,)
# ax31.set_title("Most active Z neuron depending on angle")
# pieLegend1 = patches.Patch(color=colors[0], label='Z1')
# pieLegend2 = patches.Patch(color=colors[1], label='Z2')
# pieLegend3 = patches.Patch(color=colors[2], label='Z3')
# pieLegend4 = patches.Patch(color=colors[3], label='Z4')
# pieLegend5 = patches.Patch(color=colors[4], label='Z5')
# pieLegend6 = patches.Patch(color=colors[5], label='Z6')
# pieLegend7 = patches.Patch(color=colors[6], label='Z7')
# pieLegend8 = patches.Patch(color=colors[7], label='Z8')
# pieLegend9 = patches.Patch(color=colors[8], label='Z9')
# pieLegend10 = patches.Patch(color=colors[9], label='Z10')
# ax31.legend(handles=[pieLegend1,pieLegend2,pieLegend3,pieLegend4,pieLegend5,pieLegend6,pieLegend7,pieLegend8,pieLegend9,pieLegend10], loc=(1.04, 0.25))
