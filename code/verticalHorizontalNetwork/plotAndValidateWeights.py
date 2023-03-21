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

plt.close("all")

# Command Center
plotWeights = True
validateVertical = False
validateHorizontal = False
validateCross = True
ATildeFactor = 10

imageSize = (29, 29)
imagePresentationDuration = 0.2
dt = 0.001 # seconds
firingRate = 20 # Hz
AfiringRate = 50

numberYNeurons = imageSize[0] * imageSize[1] * 2
numberZNeurons = 10
numberANeurons = 2
tauInh = 0.005
sigma = 0.01 
tauRise = 0.001
tauDecay = 0.015

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown' ,'black']
verticalLineColors = []
horizontalLineColors = []

weights = np.load("c20_3_" + ATildeFactor + "ATilde_YZWeights.npy")
priorWeights = np.load("c20_3_" + ATildeFactor + "ATilde_AZWeights.npy")
# plot all weights
if plotWeights:
  for z in range(np.shape(weights)[1]):
    plt.figure()
    plt.title("Weights of Z" + str(z+1) + " to all Y")
    wYZ = weights[0::2, z]
    wYZ = wYZ.reshape((29, 29))
    plt.imshow(wYZ, cmap='gray')
  for a in range(np.shape(priorWeights)[0]):
    plt.figure()
    plt.title("Weights of A" + str(a+1) + " to all Z")
    wAZ = priorWeights[a].reshape((10, 1))
    plt.imshow(wAZ, cmap='gray')
    
  
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
  pickle.dump(fig_object, open('verticalLines.pickle','wb'))
  plt.savefig('verticalLines.png')
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  fig_object = plt.figure()
  plt.plot(distinctZFiredHistory)
  plt.title("Number of distinct Z neurons spiking for vertical lines")
  plt.ylabel("Number of distinct Z neurons spiking")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open('verticalDistinctZ.pickle','wb'))
  plt.savefig('verticalDistinctZ.png')
  
  
  fig_object = plt.figure()
  for i in range(0, len(ZSpikeHistory[0])):
    plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  plt.title("Z Spikes for vertical lines")
  plt.ylabel("Z Neuron")
  plt.xlabel("t [s]")
  pickle.dump(fig_object, open('verticalZSpikes.pickle','wb'))
  plt.savefig('verticalZSpikes.png')
    
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  plt.figure()
  plt.plot(averageZFiredHistory)
  plt.title("Certainty of network for vertical lines")
  plt.ylabel("Homogeneity of Z spikes")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open('verticalAverageZ.pickle','wb'))  
  plt.savefig('verticalAverageZ.png')

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
  pickle.dump(fig_object, open('horizontalLines.pickle','wb'))
  plt.savefig('horizontalLines.png')
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  fig_object = plt.figure()
  plt.plot(distinctZFiredHistory)
  plt.title("Number of distinct Z neurons spiking for horizontal lines")
  plt.ylabel("Number of distinct Z neurons spiking")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open('horizontalDistinctZ.pickle','wb'))
  plt.savefig('horizontalDistinctZ.png')
  
  fig_object = plt.figure()
  for i in range(0, len(ZSpikeHistory[0])):
    plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  plt.title("Z Spikes for horizontal lines")
  plt.ylabel("Z Neuron")
  plt.xlabel("t [s]")
  pickle.dump(fig_object, open('horizontalZSpikes.pickle','wb'))
  plt.savefig('horizontalZSpikes.png')
  
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  plt.figure()
  plt.plot(averageZFiredHistory)
  plt.title("Certainty of network for horizontal lines")
  plt.ylabel("Homogeneity of Z spikes")
  plt.xlabel("Image shown")
  pickle.dump(fig_object, open('horizontalAverageZ.pickle','wb'))
  plt.savefig('horizontalAverageZ.png')

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
  plt.imshow(image, cmap='gray')
  images[0].append(image)
  images[1].append(position)
  images[2].append(prior)
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
    
# plot vertical lines 
xPosition = np.arange(imageSize[prior])
fig_object = plt.figure()
plt.bar(xPosition, imageSize[prior], align='edge', width=1.0, color=verticalLineColors)
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
pickle.dump(fig_object, open('crossLines.pickle','wb'))
plt.savefig('crossLines.png')

# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
distinctZFiredHistory.pop(0)
fig_object = plt.figure()
plt.plot(distinctZFiredHistory)
plt.title("Number of distinct Z neurons spiking for a cross")
plt.ylabel("Number of distinct Z neurons spiking")
plt.xlabel("Image shown")
pickle.dump(fig_object, open('crossDistinctZ.pickle','wb'))
plt.savefig('crossDistinctZ.png')


fig_object = plt.figure()
for i in range(0, len(ZSpikeHistory[0])):
  plt.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
plt.title("Z Spikes for a cross image")
plt.ylabel("Z Neuron")
plt.xlabel("t [s]")
pickle.dump(fig_object, open('crossZSpikes.pickle','wb'))
plt.savefig('crossZSpikes.png')
  
# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
plt.figure()
plt.plot(averageZFiredHistory)
plt.title("Certainty of network for a cross image")
plt.ylabel("Homogeneity of Z spikes")
plt.xlabel("Image shown")
pickle.dump(fig_object, open('crossAverageZ.pickle','wb'))  
plt.savefig('cross.png')

