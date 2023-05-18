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
import random

plt.close("all")

# Command Center
plotWeights = False
validateVertical = False
validateHorizontal = False
validateCross = False
showImpactOfVariablePriorOnCross = True
ATildeFactor = 1

imageSize = (35, 35)
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz
AfiringRate = 200

numberYNeurons = imageSize[0] * imageSize[1] * 2
numberZNeurons = 10
numberANeurons = 20
sigma = 0.01 
tauRise = 0.001
tauDecay = 0.015
RStar = 200 # Hz; total output firing rate

directoryPath =  "TEST" + str(numberANeurons)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']
verticalLineColors = []
horizontalLineColors = []

# weights = np.load("c20_eta3_" + "ATildeFactor" + str(ATildeFactor) + "_YZWeights.npy")
# priorWeights = np.load("c20_eta3_" + "ATildeFactor" + str(ATildeFactor) + "_AZWeights.npy")
weights = np.load("c20_eta3_" + "ATildeFactor" + str(ATildeFactor) + "_YZWeights"  + "_numberPriorNeurons" + str(numberANeurons) + ".npy")
priorWeights = np.load("c20_eta3_" + "ATildeFactor" + str(ATildeFactor) + "_AZWeights"  + "_numberPriorNeurons" + str(numberANeurons) + ".npy")

if plotWeights:
# plot all weights
  for z in range(np.shape(weights)[1]):
    plt.figure()
    plt.title("Weights of y" + str(z+1) + " to all input neurons")
    wYZ = weights[0::2, z]
    wYZ = wYZ.reshape(imageSize)
    plt.imshow(wYZ, origin='lower', cmap='gray')
    plt.savefig(directoryPath + '/weight' + str(z+1) + '.png')
  for a in range(np.shape(priorWeights)[0]):
    plt.figure()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.title("Weights of z" + str(a+1) + " to all output neurons")
    wAZ = priorWeights[a].reshape((10, 1))
    plt.imshow(wAZ, origin='lower', cmap='gray')
    plt.savefig(directoryPath + '/priorWeight' + str(a+1) + '.png')

if validateVertical:
  # validate for vertical images 
  
  # for each position generate a line image and see which Z neuron spikes most and 
  # how many distinct Z spike
  distinctZFiredHistory = []
  distinctZFired = []
  averageZFired = []
  averageZFiredHistory = []
  ZSpikeHistory = [[],[]]
  
  for positionIterator in range(0, imageSize[0]):
    YSpikes = [[],[]]
    ZSpikes = [[],[]]
    ASpikes = [[],[]]
    images = [[],[],[],[]]
    image, position, prior, orientation = dataGenerator.generateVerticalLineImage(positionIterator, imageSize)  
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

      # generate A Spikes for this step
      if prior == 0:
        for priorIterator in range (0, int(numberANeurons/2)):
          if poissonGenerator.doesNeuronFire(AfiringRate, dt):
            ASpikes[0].append(t)
            ASpikes[1].append(priorIterator)
      elif prior == 1:
        for priorIterator in range (int(numberANeurons/2), numberANeurons):
          if poissonGenerator.doesNeuronFire(AfiringRate, dt):
            ASpikes[0].append(t)
            ASpikes[1].append(priorIterator)
          
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
        ZSpikeHistory[0].append(ZNeuronWinner)
        ZSpikes[1].append(t)
        averageZFired.append(ZNeuronWinner)
        ZSpikeHistory[1].append(t + positionIterator * imagePresentationDuration)
        # append ID of Z if this Z has not fired yet in this imagePresentationDuration
        if not distinctZFired.count(ZNeuronWinner):
          distinctZFired.append(ZNeuronWinner)
    # determine distinctZ
    
    # calc which Z fired the most for this position
    whichZFired = np.zeros(numberZNeurons)
    for i in range(len(images[0])):
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
      verticalLineColors.append('white')
    else:
      verticalLineColors.append(colors[winnerID])
    
    print("Finished simulation of vertical position " + str(positionIterator))
    
  # plot vertical lines 
  xPosition = np.arange(imageSize[0])
  fig_object = plt.figure()
  plt.bar(xPosition, imageSize[0], align='edge', width=1.0, color=verticalLineColors)
  plt.title("Most active Z neuron depending on position and orientation ")
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
  pickle.dump(fig_object, open(directoryPath + '/verticalLines.pickle','wb'))
  plt.savefig(directoryPath + '/verticalLines.png')
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  fig_object = plt.figure()
  plt.plot(distinctZFiredHistory)
  plt.title("Number of distinct Z neurons spiking for vertical lines")
  plt.ylabel("Number of distinct Z neurons spiking")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open(directoryPath + '/verticalDistinctZ.pickle','wb'))
  plt.savefig(directoryPath + '/verticalDistinctZ.png')
  
  
  fig_object = plt.figure()
  for i in range(0, len(ZSpikeHistory[0])):
    plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  plt.title("Z Spikes for vertical lines")
  plt.ylabel("Z Neuron")
  plt.xlabel("t [s]")
  pickle.dump(fig_object, open(directoryPath + '/verticalZSpikes.pickle','wb'))
  plt.savefig(directoryPath + '/verticalZSpikes.png')
    
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  plt.figure()
  plt.plot(averageZFiredHistory)
  plt.title("Certainty of network for vertical lines")
  plt.ylabel("Homogeneity of Z spikes")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open(directoryPath + '/verticalAverageZ.pickle','wb'))  
  plt.savefig(directoryPath + '/verticalAverageZ.png')

if validateHorizontal:
  # validate for horizontal images 
  
  # for each position generate a line image and see which Z neuron spikes most and 
  # how many distinct Z spike
  distinctZFiredHistory = []
  distinctZFired = []
  averageZFired = []
  averageZFiredHistory = []
  ZSpikeHistory = [[],[]]
  
  for positionIterator in range(0, imageSize[1]):
    YSpikes = [[],[]]
    ZSpikes = [[],[]]
    ASpikes = [[],[]]
    images = [[],[],[],[]]
    image, position, prior, orientation = dataGenerator.generateHorizontalLineImage(positionIterator, imageSize)  
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
           
      # generate A Spikes for this step
      if prior == 0:
        for priorIterator in range (0, int(numberANeurons/2)):
          if poissonGenerator.doesNeuronFire(AfiringRate, dt):
            ASpikes[0].append(t)
            ASpikes[1].append(priorIterator)
      elif prior == 1:
        for priorIterator in range (int(numberANeurons/2), numberANeurons):
          if poissonGenerator.doesNeuronFire(AfiringRate, dt):
            ASpikes[0].append(t)
            ASpikes[1].append(priorIterator)       
          
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
        ZSpikeHistory[0].append(ZNeuronWinner)
        ZSpikes[1].append(t)
        averageZFired.append(ZNeuronWinner)
        ZSpikeHistory[1].append(t + positionIterator * imagePresentationDuration)
        # append ID of Z if this Z has not fired yet in this imagePresentationDuration
        if not distinctZFired.count(ZNeuronWinner):
          distinctZFired.append(ZNeuronWinner)
    # determine distinctZ
    
    # calc which Z fired the most for this position
    whichZFired = np.zeros(numberZNeurons)
    for i in range(len(images[0])):
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
      horizontalLineColors.append('white')
    else:
      horizontalLineColors.append(colors[winnerID])
    
    print("Finished simulation of horizontal position " + str(positionIterator))
    
  # plot vertical lines 
  yPosition = np.arange(imageSize[1])
  fig_object = plt.figure()
  plt.barh(yPosition, imageSize[1], align='edge', height=1.0, color=horizontalLineColors)
  plt.title("Most active Z neuron depending on position and orientation ")
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
  pickle.dump(fig_object, open(directoryPath + '/horizontalLines.pickle','wb'))
  plt.savefig(directoryPath + '/horizontalLines.png')
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  fig_object = plt.figure()
  plt.plot(distinctZFiredHistory)
  plt.title("Number of distinct Z neurons spiking for horizontal lines")
  plt.ylabel("Number of distinct Z neurons spiking")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open(directoryPath + '/horizontalDistinctZ.pickle','wb'))
  plt.savefig(directoryPath + '/horizontalDistinctZ.png')
  
  fig_object = plt.figure()
  for i in range(0, len(ZSpikeHistory[0])):
    plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  plt.title("Z Spikes for horizontal lines")
  plt.ylabel("Z Neuron")
  plt.xlabel("t [s]")
  pickle.dump(fig_object, open(directoryPath + '/horizontalZSpikes.pickle','wb'))
  plt.savefig(directoryPath + '/horizontalZSpikes.png')
  
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  plt.figure()
  plt.plot(averageZFiredHistory)
  plt.title("Certainty of network for horizontal lines")
  plt.ylabel("Homogeneity of Z spikes")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open(directoryPath + '/horizontalAverageZ.pickle','wb'))
  plt.savefig(directoryPath + '/horizontalAverageZ.png')

if validateCross:
  # validate for cross images 
  
  # VERTICAL 
  # for each position generate a line image and see which Z neuron spikes most and 
  # how many distinct Z spike
  distinctZFiredHistory = []
  distinctZFired = []
  averageZFired = []
  averageZFiredHistory = []
  ZSpikeHistory = [[],[]]
  

  YSpikes = [[],[]]
  ZSpikes = [[],[]]
  ASpikes = [[],[]]
  images = [[],[],[],[]]
  image, position, prior = dataGenerator.generateRandomCrossLineImage() 
  plt.figure()
  plt.imshow(image, origin='lower', cmap='gray')
  plt.savefig(directoryPath + '/crossImage.png')
  images[0].append(image)
  images[1].append(position)
  images[2].append(prior)
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
         
    # generate A Spikes for this step
    if prior == 0:
      for priorIterator in range (0, int(numberANeurons/2)):
        if poissonGenerator.doesNeuronFire(AfiringRate, dt):
          ASpikes[0].append(t)
          ASpikes[1].append(priorIterator)
    elif prior == 1:
      for priorIterator in range (int(numberANeurons/2), numberANeurons):
        if poissonGenerator.doesNeuronFire(AfiringRate, dt):
          ASpikes[0].append(t)
          ASpikes[1].append(priorIterator)    
        
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
      ZSpikeHistory[0].append(ZNeuronWinner)
      ZSpikes[1].append(t)
      averageZFired.append(ZNeuronWinner)
      # ZSpikeHistory[1].append(t + positionIterator * imagePresentationDuration)
      # !!! Swapped changed the above and below line, bc only one iteration right now
      ZSpikeHistory[1].append(t)
      # append ID of Z if this Z has not fired yet in this imagePresentationDuration
      if not distinctZFired.count(ZNeuronWinner):
        distinctZFired.append(ZNeuronWinner)
  
  fig_object = plt.figure()
  for i in range(0, len(ZSpikeHistory[0])):
    plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  plt.title("Z Spikes for a cross image with prior = " + str(prior))
  plt.ylabel("Z Neuron")
  plt.xlabel("t [s]")
  pickle.dump(fig_object, open(directoryPath + '/crossZSpikes.pickle','wb'))
  plt.savefig(directoryPath + '/crossZSpikes.png')

if showImpactOfVariablePriorOnCross:
  # validate for cross images with variable prior 
  
  images = [[],[],[],[]]
  image, xPosition, yPosition, prior = dataGenerator.generateCrossLineImageAtPosition(0, 3, 32, imageSize) 
  plt.figure()
  plt.imshow(image, origin='lower', cmap='gray')
  plt.savefig(directoryPath + '/crossImageForVariablePrior.png')
  images[0].append(image)
  images[1].append(xPosition)
  images[2].append(prior)
  encodedImage = dataEncoder.encodeImage(image)
  
  winnerID = np.full((AfiringRate, 1), math.inf)
  secondWinnerID = np.full((AfiringRate, 1), math.inf)
  maxSpikesOfWinner = np.full((AfiringRate, 1), 0)
  maxSpikesOfSecondWinner = np.full((AfiringRate, 1), 0)

  for currentPrior in range(0, AfiringRate):
    # for each Prior calc 2 most active Y Neurons and plot their firing frequency vs current prior
    distinctZFiredHistory = []
    distinctZFired = []
    averageZFired = []
    averageZFiredHistory = []
    ZSpikeHistory = [[],[]]
    UHistory = [[],[],[],[],[],[],[],[],[],[]]
    
    YSpikes = [[],[]]
    ZSpikes = [[],[]]
    ASpikes = [[],[]]
  
    distinctZFiredHistory.append(len(distinctZFired))
    distinctZFired = []
    
    if averageZFired:
      mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
      amountMostSpikingZ = averageZFired.count(mostSpikingZ)
      averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
      averageZFired = []
    
    ASpikeHistory = []
    
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
           
      # generate A Spikes for this step... both groups of Z fire depending on
      # current prior
      for priorIterator in range (0, int(numberANeurons/2)):
        if poissonGenerator.doesNeuronFire(currentPrior, dt):
          ASpikes[0].append(t)
          ASpikes[1].append(priorIterator)
      for priorIterator in range (int(numberANeurons/2), numberANeurons):
        if poissonGenerator.doesNeuronFire(AfiringRate - currentPrior, dt):
          ASpikes[0].append(t)
          ASpikes[1].append(priorIterator)           
      
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
      
      for uValue in range(numberZNeurons):
        UHistory[uValue].append(U[uValue])

        
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
        # ZSpikeHistory[1].append(t + positionIterator * imagePresentationDuration)
        # !!! Swapped changed the above and below line, bc only one iteration right now
        ZSpikeHistory[1].append(t)
        # append ID of Z if this Z has not fired yet in this imagePresentationDuration
        if not distinctZFired.count(ZNeuronWinner):
          distinctZFired.append(ZNeuronWinner)
    
    fig_object = plt.figure()
    for i in range(0, len(ZSpikeHistory[0])):
      plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
    plt.title("Output spikes with prior = " + str(currentPrior))
    plt.ylabel("Y Neuron")
    plt.xlabel("t [s]")
    pickle.dump(fig_object, open(directoryPath + '/crossZSpikes_prior' + str(currentPrior) + '.pickle','wb'))
    plt.savefig(directoryPath + '/crossZSpikes' + str(currentPrior) + '.png')
    
    # calc which Z fired the most and second most for this prior
    whichZFired = np.zeros(numberZNeurons)
    for i in range(len(images[0])):
      # iterate over all zSpikes and increment the Z that fired
      for j in range(len(ZSpikes[0])):
        whichZFired[ZSpikes[0][j]] += 1
    for j in range(numberZNeurons):
      if whichZFired[j] > maxSpikesOfWinner[currentPrior,0]:
        secondWinnerID[currentPrior,0] = winnerID[currentPrior,0]
        maxSpikesOfSecondWinner[currentPrior,0] = maxSpikesOfWinner[currentPrior,0]
        winnerID[currentPrior,0] = j
        maxSpikesOfWinner[currentPrior,0] = whichZFired[j]
      elif whichZFired[j] > maxSpikesOfSecondWinner[currentPrior,0]:
        secondWinnerID[currentPrior,0] = j
        maxSpikesOfSecondWinner[currentPrior,0] = whichZFired[j]
        
    fig_object = plt.figure()
    for k in range(numberZNeurons):  
      plt.plot(UHistory[k], color=colors[k])
    plt.title("Membrane potentials")
    plt.ylabel("U")
    plt.xlabel("Time [ms]")
    # plt.legend()
    pickle.dump(fig_object, open(directoryPath + '/membranePotentials' + str(currentPrior) + '.pickle','wb'))
    plt.savefig(directoryPath + '/membranePotentials' + str(currentPrior) + '.png')   
    print("Finished simulation of prior " + str(currentPrior))
  
  # plot frequencies of 2 most spiking neurons over prior neuron firing frequency
  
  # determine the IDs of the 2 Y neurons that spiked most
  IdOfY0 = math.inf
  IdOfY1 = math.inf
  maxWinner0 = 0
  maxWinner1 = 0
  for i in range(0,10):  
    if np.count_nonzero(winnerID == i) > maxWinner0:
      IdOfY1 = IdOfY0
      IdOfY0 = i
    elif np.count_nonzero(winnerID == i) > maxWinner1:
      IdOfY1 = i
  
  firingRateY0 = []
  firingRateY1 = []
  # calc firing rates
  for i in range(0, AfiringRate):
    winnerIDNotPresent = False
    secondWinnerIDNotPresent = False
    # check which neuron is first and second winner
    if winnerID[i, 0] == IdOfY0:
      firingRateY0.append(maxSpikesOfWinner[i, 0] / imagePresentationDuration)
    elif winnerID[i, 0] == IdOfY1:
      firingRateY1.append(maxSpikesOfWinner[i, 0] / imagePresentationDuration)
    else:
      winnerIDNotPresent = True
    if secondWinnerID[i, 0] == IdOfY0:
      firingRateY0.append(maxSpikesOfSecondWinner[i, 0] / imagePresentationDuration)
    elif secondWinnerID[i, 0] == IdOfY1:
      firingRateY1.append(maxSpikesOfSecondWinner[i, 0] / imagePresentationDuration)
    else:
      secondWinnerIDNotPresent = True
    if winnerIDNotPresent and secondWinnerIDNotPresent:
      firingRateY0.append(0)
      firingRateY1.append(0)
    elif winnerIDNotPresent or secondWinnerIDNotPresent:
      if len(firingRateY0) > len(firingRateY1):
        firingRateY1.append(0)
      else:
        firingRateY0.append(0) 

  fig_object = plt.figure() 
  xPositionPlt = np.arange(AfiringRate)
  plt.plot(xPositionPlt, firingRateY0, color=colors[IdOfY0])
  plt.plot(xPositionPlt, firingRateY1, color=colors[IdOfY1])
  plt.title("Firing frequency of 2 most active Y neurons")
  plt.ylabel("Y firing frequency [Hz]")
  plt.xlabel("z0 firing frequency [Hz]")
  # plt.legend()
  pickle.dump(fig_object, open(directoryPath + '/YFrequency_prior' + str(currentPrior) + '.pickle','wb'))
  plt.savefig(directoryPath + '/YFrequency_prior' + str(currentPrior) + '.png')   

         
