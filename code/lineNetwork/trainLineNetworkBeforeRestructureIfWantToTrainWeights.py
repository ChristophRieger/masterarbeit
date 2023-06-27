# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:55:32 2022

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')
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

simulationTime = 200
imageSize = (1, 9)
imagePresentationDuration = 2
dt = 0.001 # seconds

firingRate = 200 # Hz; Input neurons yn should spike with 20Hz => firingRate (Lambda) = 20/second
AfiringRate = 200
numberXNeurons = imageSize[0] * imageSize[1] # 1 neurons per pixel (one for black) # input
numberYNeurons = 4 # output
numberZNeurons = 4 # prior

sigma = 0.01 # time frame in which spikes count as before output spike
c = 20 # 10 seems good, scales weights to 0 ... 1
tauRise = 0.001
tauDecay = 0.015
learningRateFactor = 3
learningRate = 10**-learningRateFactor
RStar = 200 # Hz; total output firing rate

XSpikes = [[],[]] # input
YSpikes = [[],[]] # output
ZSpikes = [[],[]] # prior

# initialize weights (for now between 0 and 1, not sure)
if loadWeights:
  weights = np.load("c20_3_YZWeights.npy")
  priorWeights = np.load("c20_3_AZWeights.npy")
else:
  weights = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]],
                        "float64")
  weights = np.log(weights)
  priorWeights = np.array([[0.9, 0.0333, 0.0333, 0.0333],
                           [0.0333, 0.9, 0.0333, 0.0333],
                           [0.0333, 0.0333, 0.9, 0.0333],
                           [0.0333, 0.0333, 0.0333, 0.9]],
                        "float64")
  priorWeights = np.log(priorWeights)

  # weights = np.full((numberXNeurons, numberYNeurons), 2, "float64")
  # priorWeights = np.full((numberZNeurons, numberYNeurons), 2, "float64")

indexOfLastYSpike = [0] * numberXNeurons
ZNeuronsRecievedYSpikes = [[],[]]
images = [[],[]]

# Metric to measure training progress
# check how many different Z neurons fired during one image
distinctZFiredHistory = []
distinctZFired = []
averageZFired = []
averageZFiredHistory = []

# image, prior = dataGenerator.generateRandom1DLineImage(imageSize)
# plt.figure()
# plt.imshow(image, cmap='gray')
# sys.exit()

# start simulation
for t in np.arange(0, simulationTime, dt):
  # generate training data every 50ms
  if abs(t - round(t / imagePresentationDuration) * imagePresentationDuration) < 1e-10:
    image, prior = dataGenerator.generateRandom1DLineImage(imageSize)
    images[0].append(image)
    images[1].append(prior)
    distinctZFiredHistory.append(len(distinctZFired))
    distinctZFired = []
    if averageZFired:
      mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
      amountMostSpikingZ = averageZFired.count(mostSpikingZ)
      averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
      averageZFired = []
    
  # generate Y Spikes for this step
  for i in range(len(image)):
    # check if the Yi is active
    if image[0][i] == 0:
     # check if Yi spiked in this timestep
     if poissonGenerator.doesNeuronFire(firingRate, dt):
       # when did it spike
       XSpikes[0].append(t)
       # which Y spiked
       XSpikes[1].append(i)
       
  # generate A Spikes for this step
  if poissonGenerator.doesNeuronFire(AfiringRate, dt):
    ZSpikes[0].append(t)
    ZSpikes[1].append(prior)

  
  # Next we have to calculate Uk
  U = np.zeros(numberYNeurons)
  # Add contribution of Y
  expiredYSpikeIDs = []
  YTilde = np.zeros(numberXNeurons)
  for i, YNeuron in enumerate(XSpikes[1]):
    # First mark all XSpikes older than sigma and do not use for calculation of Uk
    if XSpikes[0][i] < t - sigma:
      expiredYSpikeIDs.append(i)
    else:
      YTilde[XSpikes[1][i]] = kernel.tilde(t, dt, XSpikes[0][i], tauRise, tauDecay)
      for k in range(numberYNeurons):
        U[k] += weights[YNeuron, k] * YTilde[XSpikes[1][i]]
  # delete all spikes that are longer ago than sigma (10ms?) from XSpikes
  for toDeleteID in sorted(expiredYSpikeIDs, reverse=True):
    del XSpikes[0][toDeleteID]
    del XSpikes[1][toDeleteID]
    
  # Add contribution of A
  ATilde = np.zeros(numberZNeurons)
  expiredASpikeIDs = []
  for i in range(len(ZSpikes[1])):
    # First mark all ZSpikes older than sigma and do not use for calculation of Uk
    if ZSpikes[0][i] < t - sigma:
      expiredASpikeIDs.append(i)
    else:
      ATilde[ZSpikes[1][i]] = kernel.tilde(t, dt, ZSpikes[0][i], tauRise, tauDecay)
      for k in range(numberYNeurons):
        U[k] += priorWeights[prior, k] * ATilde[ZSpikes[1][i]]
  # delete all spikes that are longer ago than sigma (10ms?) from ZSpikes
  for toDeleteID in sorted(expiredASpikeIDs, reverse=True):
    del ZSpikes[0][toDeleteID]
    del ZSpikes[1][toDeleteID]


  # calculate current Inhibition signal
  inhTMP = 0
  for i in range(numberYNeurons):
    inhTMP += np.exp(U[i])
  Iinh = - np.log(RStar) + np.log(inhTMP)
    
  # calc instantaneous fire rate for each Z Neuron for this time step
  r = np.zeros(numberYNeurons)
  ZNeuronsThatWantToFire = []
  ZNeuronWantsToFireAtTime = []
  # ZNeuronFireFactors is used to choose between multiple Z firing in this timestep
  ZNeuronFireFactors = []
  for k in range(numberYNeurons):
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
    YSpikes[0].append(ZNeuronWinner)
    YSpikes[1].append(t)
    averageZFired.append(ZNeuronWinner)
    # append ID of Z if this Z has not fired yet in this imagePresentationDuration
    if not distinctZFired.count(ZNeuronWinner):
      distinctZFired.append(ZNeuronWinner)
    # update weights of all Y to ZThatFired
    # do not update weights in this experiment for now we want to analyze the mathematically determined weights
    weights = neuronFunctions.updateWeights(YTilde, weights, ZNeuronWinner, c, learningRate)
    priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronWinner, c, learningRate)
    
  if (t) % 1 == 0:
    print("Finished simulation of t= " + str(t))

directoryPath =  "c" + str(c) + "_eta" + str(learningRateFactor) + "_numberPriorNeurons" + str(numberZNeurons)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)
np.save(directoryPath + "/c" + str(c) + "_eta" + str(learningRateFactor) + "_YZWeights"  + "_numberPriorNeurons" + str(numberZNeurons) + ".npy", weights)
np.save(directoryPath + "/c" + str(c) + "_eta" + str(learningRateFactor) + "_AZWeights"  + "_numberPriorNeurons" + str(numberZNeurons) + ".npy", priorWeights)
  
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
for z in range(int(numberYNeurons/2)):
  # 4 = linethickness/2 aufgerundet
  # TODO due to int() we are 1 pixel off in plots... fix pls
  image = dataGenerator.generateHorizontalLineImage(int(3.5 + z * (imageSize[0]) / (numberYNeurons/2)), imageSize)
  eval("ax1" + str(z) + ".imshow(image[0], cmap='gray')")
  eval("ax1" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")
  w = weights[0::2, z]
  w = w.reshape((imageSize[0], imageSize[1]))
  eval("ax2" + str(z) + ".imshow(w, cmap='gray')")
  eval("ax2" + str(z) + ".set_title('$w_{' + str(z+1) + 'n}$', pad=5)")
  eval("ax2" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")

for z in range(int(numberYNeurons / 2), int(numberYNeurons)):
  # 4 = linethickness/2 aufgerundet
  image = dataGenerator.generateVerticalLineImage(int(3.5 + (z - 5) * (imageSize[1]) / (numberYNeurons/2)), imageSize)
  eval("ax1" + str(z) + ".imshow(image[0], cmap='gray')")
  eval("ax1" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")
  
  w = weights[0::2, z]
  w = w.reshape((imageSize[0], imageSize[1]))
  eval("ax2" + str(z) + ".imshow(w, cmap='gray')")
  eval("ax2" + str(z) + ".set_title('$w_{' + str(z+1) + 'n}$', pad=5)")
  eval("ax2" + str(z) + ".tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelright=False, labelleft=False)")

# plot the first 120 output spikes
for i in range(0, 120):
  ax31.vlines(YSpikes[1][i], ymin=YSpikes[0][i] + 1 - 0.5, ymax=YSpikes[0][i] + 1 + 0.5, color=colors[YSpikes[0][i]], linewidth=0.5)
ax31.axvline(x=0, color="red", linestyle="dashed")
ax31.axvline(x=0.2, color="red", linestyle="dashed")
ax31.axvline(x=0.4, color="red", linestyle="dashed")
ax31.axvline(x=0.6, color="red", linestyle="dashed")
ax31.set_title("Output before learning")
ax31.set_ylabel("Output neuron")
ax31.set_xlabel("Time [s]")

# plot the last 120 Z spikes
for i in range(len(YSpikes[0]) - len(YSpikes[0][-120:]), len(YSpikes[0])):
  ax32.vlines(YSpikes[1][i], ymin=YSpikes[0][i] + 1 - 0.5, ymax=YSpikes[0][i] + 1 + 0.5, color=colors[YSpikes[0][i]], linewidth=0.5)
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
plt.savefig(directoryPath + "/trainingPlot.png")
plt.savefig(directoryPath + "/trainingPlot.jpg")    








# plot all images... 
# for imageToPlot in images[0]:
#   plt.figure()
#   plt.imshow(imageToPlot, cmap='gray')

# calc which Z fired the most for horizontal position
positionAndWhichZFiredHorizontally = np.zeros([imageSize[0], numberYNeurons])
for i in range(len(images[0])):
  # only take horizontal images
  if images[3][i] == 1:
    # check which position the image was
    for position in range(imageSize[0]):
      if images[1][i] > position and images[1][i] <= position + 1:
        currentPosition = position
    for j in range(len(YSpikes[0])):
      # get all spikes between t and t+imagePresentationDuration
      # !!! maybe save last index to not always start from 0 like above for YNeuronspikes
      if YSpikes[1][j] > i * imagePresentationDuration and YSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
        positionAndWhichZFiredHorizontally[currentPosition, YSpikes[0][j]] += 1
# determine Z neuron that fired the most for each horizontal position
horizontalLineColors = []
for i in range(imageSize[0]):
  winnerID = math.inf
  maxSpikes = 0
  for j in range(numberYNeurons):
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
positionAndWhichZFiredVertically = np.zeros([imageSize[1], numberYNeurons])
for i in range(len(images[0])):
  # only take horizontal images
  if images[3][i] == 0:
    # check which position the image was
    for position in range(imageSize[1]):
      if images[1][i] > position and images[1][i] <= position + 1:
        currentPosition = position
    for j in range(len(YSpikes[0])):
      # get all spikes between t and t+imagePresentationDuration
      # !!! maybe save last index to not always start from 0 like above for YNeuronspikes
      if YSpikes[1][j] > i * imagePresentationDuration and YSpikes[1][j] < i * imagePresentationDuration + imagePresentationDuration:
        positionAndWhichZFiredVertically[currentPosition, YSpikes[0][j]] += 1
# determine Z neuron that fired the most for each vertical position
verticalLineColors = []
for i in range(imageSize[1]):
  winnerID = math.inf
  maxSpikes = 0
  for j in range(numberYNeurons):
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