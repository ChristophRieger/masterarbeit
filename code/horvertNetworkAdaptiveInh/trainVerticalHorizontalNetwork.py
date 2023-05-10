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

plt.close("all")
# # generate training data
# imageSize = (28, 28)
# image = dataGenerator.generateRandomlyOrientedLineImage()
# # plt.imshow(image, cmap='gray')
# encodedImage = dataEncoder.encodeImage(image)

# took 35, lineWidth (7) * outputneurons / 2
imageSize = (35, 35)
simulationTime = 800 # seconds
# legi suggested to increase this from 0.05 to 0.2, works better
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz; Input neurons yn should spike with 20Hz => firingRate (Lambda) = 20/second
AfiringRate = 200
numberYNeurons = imageSize[0] * imageSize[1] * 2 # 2 neurons per pixel (one for black, one for white)
numberZNeurons = 10
numberANeurons = 2

sigma = 0.01 # time frame in which spikes count as before output spike
c = 20 # 10 seems good, scales weights to 0 ... 1
cPrior = 20
tauRise = 0.001
tauDecay = 0.015
learningRateFactor = 3
learningRate = 10**-learningRateFactor
ATildeFactor = 1
RStar = 200 # Hz; total output firing rate

YSpikes = [[],[]]
ZSpikes = [[],[]]
ASpikes = [[],[]]

# initialize weights (for now between 0 and 1, not sure)
if loadWeights:
  weights = np.load("c20_3_" + ATildeFactor + "ATilde_YZWeights.npy")
  priorWeights = np.load("c20_3_" + ATildeFactor + "ATilde_AZWeights.npy")
else:
  weights = np.full((numberYNeurons, numberZNeurons), 2, "float64")
  priorWeights = np.full((numberANeurons, numberZNeurons), 2, "float64")

indexOfLastYSpike = [0] * numberYNeurons
ZNeuronsRecievedYSpikes = [[],[]]
images = [[],[],[],[]]

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
    if random.random() > 0.5:
      image, position, prior, orientation = dataGenerator.generateRandomVerticalLineImage(imageSize)
    else:
      image, position, prior, orientation = dataGenerator.generateRandomHorizontalLineImage(imageSize)
    images[0].append(image)
    images[1].append(position)
    images[2].append(prior)
    images[3].append(orientation)
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
       
  # generate A Spikes for this step
  if prior == 0:
    if poissonGenerator.doesNeuronFire(AfiringRate, dt):
      ASpikes[0].append(t)
      ASpikes[1].append(0)
  elif prior == 1:
    if poissonGenerator.doesNeuronFire(AfiringRate, dt):
      ASpikes[0].append(t)
      ASpikes[1].append(1)
  
  # Next we have to calculate Uk
  U = np.zeros(numberZNeurons)
  # Add contribution of Y
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
    
  # Add contribution of A
  ATilde = np.zeros(numberANeurons)
  expiredASpikeIDs = []
  for i, ANeuron in enumerate(ASpikes[1]):
    # First mark all ASpikes older than sigma and do not use for calculation of Uk
    if ASpikes[0][i] < t - sigma:
      expiredASpikeIDs.append(i)
    else:
      ATilde[ASpikes[1][i]] = ATildeFactor * kernel.tilde(t, dt, ASpikes[0][i], tauRise, tauDecay)
      for k in range(numberZNeurons):
        U[k] += priorWeights[ANeuron, k] * ATilde[ASpikes[1][i]]
  # delete all spikes that are longer ago than sigma (10ms?) from ASpikes
  for toDeleteID in sorted(expiredASpikeIDs, reverse=True):
    del ASpikes[0][toDeleteID]
    del ASpikes[1][toDeleteID]


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
    priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronWinner, cPrior, learningRate)
    
  if (t) % 1 == 0:
    print("Finished simulation of t= " + str(t))

directoryPath =  "c" + str(c) + "_eta" + str(learningRateFactor) + "_ATildeFactor" + str(ATildeFactor)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)
np.save(directoryPath + "/c" + str(c) + "_eta" + str(learningRateFactor) + "_ATildeFactor" + str(ATildeFactor) + "_YZWeights.npy", weights)
np.save(directoryPath + "/c" + str(cPrior) + "_eta" + str(learningRateFactor) + "_ATildeFactor" + str(ATildeFactor) + "_AZWeights.npy", priorWeights)
  
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']

fig = plt.figure()
gs = fig.add_gridspec(6, 10, hspace=1)
gs2 = fig.add_gridspec(6, 2, wspace=0.4, hspace=25)
# hspace seems to be capped and doesnt really influence the spacing anymore. 
# but the .svg seems fine anyway, solve only if figure is not good enough.
gs3 = fig.add_gridspec(6, 2, wspace=0.4, hspace=400)


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

ax41 = fig.add_subplot(gs3[4:6, 0])
ax42 = fig.add_subplot(gs3[4:6, 1])

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

ax_fourthRowFirstColumn = fig.add_subplot(gs3[4:6, 0])
ax_fourthRowFirstColumn.axis('off')
ax_fourthRowFirstColumn.set_title('E', loc="left", x=-0.1, fontsize=16.0, fontweight='semibold')
ax_fourthRowSecondColumn = fig.add_subplot(gs3[4:6, 1])
ax_fourthRowSecondColumn.axis('off')
ax_fourthRowSecondColumn.set_title('F', loc="left", x=-0.1, fontsize=16.0, fontweight='semibold')

# plot training data examples and weights
for z in range(int(numberZNeurons/2)):
  # 4 = linethickness/2 aufgerundet
  # TODO due to int() we are 1 pixel off in plots... fix pls
  image = dataGenerator.generateHorizontalLineImage(int(3.5 + z * (imageSize[0]) / (numberZNeurons/2)), imageSize)
  eval("ax1" + str(z) + ".imshow(image[0], cmap='gray')")
  eval("ax1" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")
  w = weights[0::2, z]
  w = w.reshape((imageSize[0], imageSize[1]))
  eval("ax2" + str(z) + ".imshow(w, cmap='gray')")
  eval("ax2" + str(z) + ".set_title('$w_{' + str(z+1) + 'n}$', pad=5)")
  eval("ax2" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")

for z in range(int(numberZNeurons / 2), int(numberZNeurons)):
  # 4 = linethickness/2 aufgerundet
  image = dataGenerator.generateVerticalLineImage(int(3.5 + (z - 5) * (imageSize[1]) / (numberZNeurons/2)), imageSize)
  eval("ax1" + str(z) + ".imshow(image[0], cmap='gray')")
  eval("ax1" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")
  
  w = weights[0::2, z]
  w = w.reshape((imageSize[0], imageSize[1]))
  eval("ax2" + str(z) + ".imshow(w, cmap='gray')")
  eval("ax2" + str(z) + ".set_title('$w_{' + str(z+1) + 'n}$', pad=5)")
  eval("ax2" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")

# plot the first 120 output spikes
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


# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
distinctZFiredHistory.pop(0)
ax41.plot(distinctZFiredHistory)
ax41.set_title("Training progress")
ax41.set_ylabel("Output neurons spiking")
ax41.set_xlabel("Image shown")

# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
ax42.plot(averageZFiredHistory)
ax42.set_title("Certainty of the network")
ax42.set_ylabel("Certainty")
ax42.set_xlabel("Image shown")

plt.show()
pickle.dump(fig, open(directoryPath + "/trainingPlot" + '.pickle','wb'))
plt.savefig(directoryPath + "/trainingPlot.svg")  








# plot all images... 
# for imageToPlot in images[0]:
#   plt.figure()
#   plt.imshow(imageToPlot, cmap='gray')

# calc which Z fired the most for horizontal position
positionAndWhichZFiredHorizontally = np.zeros([imageSize[0], numberZNeurons])
for i in range(len(images[0])):
  # only take horizontal images
  if images[3][i] == 1:
    # check which position the image was
    for position in range(imageSize[0]):
      if images[1][i] > position and images[1][i] <= position + 1:
        currentPosition = position
    for j in range(len(ZSpikes[0])):
      # get all spikes between t and t+imagePresentationDuration
      # !!! maybe save last index to not always start from 0 like above for YNeuronspikes
      if ZSpikes[1][j] > i * imagePresentationDuration and ZSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
        positionAndWhichZFiredHorizontally[currentPosition, ZSpikes[0][j]] += 1
# determine Z neuron that fired the most for each horizontal position
horizontalLineColors = []
for i in range(imageSize[0]):
  winnerID = math.inf
  maxSpikes = 0
  for j in range(numberZNeurons):
    if positionAndWhichZFiredHorizontally[i][j] > maxSpikes:
      winnerID = j
      maxSpikes = positionAndWhichZFiredHorizontally[i][j]
  # code winnerID of Z neuron to its color used above
  if winnerID == math.inf:
    horizontalLineColors.append('white')
  else:
    horizontalLineColors.append(colors[winnerID])
# plot horizontal lines 
yPosition = np.arange(imageSize[1])
plt.figure()
plt.barh(yPosition, imageSize[1], align='edge', height=1.0, color=horizontalLineColors)
plt.title("Most active output neuron depending on position and orientation ")
plt.xlabel("width [px]")
plt.ylabel("height [px]")
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
plt.tight_layout()
plt.savefig(directoryPath + "/horizontalLines.png")        
    
# calc which Z fired the most for vertical position
positionAndWhichZFiredVertically = np.zeros([imageSize[1], numberZNeurons])
for i in range(len(images[0])):
  # only take horizontal images
  if images[3][i] == 0:
    # check which position the image was
    for position in range(imageSize[1]):
      if images[1][i] > position and images[1][i] <= position + 1:
        currentPosition = position
    for j in range(len(ZSpikes[0])):
      # get all spikes between t and t+imagePresentationDuration
      # !!! maybe save last index to not always start from 0 like above for YNeuronspikes
      if ZSpikes[1][j] > i * imagePresentationDuration and ZSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
        positionAndWhichZFiredVertically[currentPosition, ZSpikes[0][j]] += 1
# determine Z neuron that fired the most for each vertical position
verticalLineColors = []
for i in range(imageSize[1]):
  winnerID = math.inf
  maxSpikes = 0
  for j in range(numberZNeurons):
    if positionAndWhichZFiredVertically[i][j] > maxSpikes:
      winnerID = j
      maxSpikes = positionAndWhichZFiredVertically[i][j]
  # code winnerID of Z neuron to its color used above
  if winnerID == math.inf:
    verticalLineColors.append('white')
  else:
    verticalLineColors.append(colors[winnerID])
# plot vertical lines 
xPosition = np.arange(imageSize[0])
plt.figure()
plt.bar(xPosition, imageSize[0], align='edge', width=1.0, color=verticalLineColors)
plt.title("Most active output neuron depending on position and orientation ")
plt.xlabel("width [px]")
plt.ylabel("height [px]")
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
plt.tight_layout()
plt.savefig(directoryPath + "/verticalLines.png")      