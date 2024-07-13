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
plotWeights = True
validateVertical = False
validateHorizontal = False
validateCross = False
showImpactOfVariablePriorOnCross = False
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

directoryPath =  "newPriorWeightPlot" + str(numberANeurons)
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
  # for z in range(np.shape(weights)[1]):
  #   plt.figure()
  #   plt.title("Weights of y" + str(z+1) + " to all input neurons")
  #   wYZ = weights[0::2, z]
  #   wYZ = wYZ.reshape(imageSize)
  #   plt.imshow(wYZ, origin='lower', cmap='gray')
  #   plt.savefig(directoryPath + '/weight' + str(z+1) + '.png')
    
    
    
  fig, axs = plt.subplots(nrows=2, ncols=int(len(priorWeights)/2), subplot_kw={'xticks': []})
  
  priorWeightCounter = 0
  for ax in axs.flat:
    priorWeightCounterString = str(priorWeightCounter+1) 
    ax.set_title("$w^p$" + "$_k$" + r'$_{{{stringToAdd}}}$'.format(stringToAdd=priorWeightCounterString), fontsize = 14)
    wAZ = priorWeights[priorWeightCounter].reshape((10, 1))
    ax.set_yticks([0,1,2,3,4,5,6,7,8,9], labels=[1,2,3,4,5,6,7,8,9,10])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Output neuron", fontsize=12)
    ax.yaxis.set_label_coords(-0.63, 0.5)

    im = ax.imshow(wAZ, origin='lower', cmap='gray')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)


    priorWeightCounter += 1
  fig.suptitle('Prior weights', fontsize=16)
  # plt.savefig(directoryPath + '/priorWeight' + str(a+1) + '.png')

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

  fig = plt.figure(figsize=[10, 8])
  # gs = fig.add_gridspec(2, 2, wspace=1, hspace=1)
  gs = fig.add_gridspec(2, 2, wspace=0.6, hspace=0.6)

  
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[0, 1])
  ax3 = fig.add_subplot(gs[1, 0])
  ax4 = fig.add_subplot(gs[1, 1])

  # plot vertical lines 
  yPosition = np.arange(imageSize[1])
  ax1.bar(yPosition, imageSize[1], align='center', width=1.0, color=verticalLineColors)
  ax1.set_title("Most active output neuron", fontsize=14)
  # ax1.set_xlabel("Width [px]", fontsize=12)
  ax1.set_xlabel("Position [px]", fontsize=12)
  ax1.tick_params(axis="x", labelsize=12)
  ax1.tick_params(axis="y", which='both', left=False, right=False, labelleft=False)
  # ax1.set_ylim([0,35])
  ax1.xaxis.set_ticks(np.arange(0,36, 5))
  pieLegend1 = patches.Patch(color=colors[0], label='y1')
  # pieLegend2 = patches.Patch(color=colors[1], label='y2')
  pieLegend3 = patches.Patch(color=colors[2], label='y3')
  # pieLegend4 = patches.Patch(color=colors[3], label='y4')
  pieLegend5 = patches.Patch(color=colors[4], label='y5')
  # pieLegend6 = patches.Patch(color=colors[5], label='y6')
  # pieLegend7 = patches.Patch(color=colors[6], label='y7')
  # pieLegend8 = patches.Patch(color=colors[7], label='y8')
  pieLegend9 = patches.Patch(color=colors[8], label='y9')
  pieLegend10 = patches.Patch(color=colors[9], label='y10')
  ax1.legend(handles=[pieLegend1,pieLegend3,pieLegend5,pieLegend9,pieLegend10], loc=(1.02, 0.25), prop={'size': 12})
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  ax3.plot(distinctZFiredHistory)
  ax3.set_title("Active output neurons", fontsize=14)
  ax3.set_ylabel("# of active output neurons", fontsize=12)
  ax3.set_xlabel("Position [px]", fontsize=12)
  ax3.set_ylim([0.8,2.2])
  ax3.xaxis.set_ticks(np.arange(0,36, 5))
  ax3.yaxis.set_ticks(np.arange(1,3, 1))
  ax3.tick_params(axis="both", labelsize=12)


  for i in range(0, len(ZSpikeHistory[0])):
    ax2.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  ax2.set_title("Output spikes", fontsize=14)
  ax2.set_ylabel("Output neuron", fontsize=12)
  ax2.set_xlabel("t [s]", fontsize=12)
  ax2.set_ylim([0,11])
  ax2.yaxis.set_ticks(np.arange(1,11, 1))
  ax2.tick_params(axis="both", labelsize=12)
 
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  ax4.plot(averageZFiredHistory)
  ax4.set_title("Relative activity of most active output neuron", fontsize=14)
  ax4.set_ylabel("Relative activity", fontsize=12)
  ax4.set_xlabel("Position [px]", fontsize=12)
  ax4.xaxis.set_ticks(np.arange(0,36, 5))
  ax4.tick_params(axis="both", labelsize=12)

  
  # plt.tight_layout()
  pickle.dump(fig, open(directoryPath + '/vertical_validation.pickle','wb'))
  plt.savefig(directoryPath + "/vertical_validation.svg") 
  plt.show()

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
    
  fig = plt.figure(figsize=[10, 8])
  # gs = fig.add_gridspec(2, 2, wspace=1, hspace=1)
  gs = fig.add_gridspec(2, 2, wspace=0.6, hspace=0.6)

  
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[0, 1])
  ax3 = fig.add_subplot(gs[1, 0])
  ax4 = fig.add_subplot(gs[1, 1])

  # plot horizontal lines 
  yPosition = np.arange(imageSize[1])
  ax1.barh(yPosition, imageSize[1], align='center', height=1.0, color=horizontalLineColors)
  ax1.set_title("Most active output neuron", fontsize=14)
  # ax1.set_xlabel("Width [px]", fontsize=12)
  ax1.set_ylabel("Position [px]", fontsize=12)
  ax1.tick_params(axis="y", labelsize=12)
  ax1.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False)
  # ax1.set_ylim([0,35])
  ax1.yaxis.set_ticks(np.arange(0,36, 5))
  # pieLegend1 = patches.Patch(color=colors[0], label='y1')
  pieLegend2 = patches.Patch(color=colors[1], label='y2')
  # pieLegend3 = patches.Patch(color=colors[2], label='y3')
  pieLegend4 = patches.Patch(color=colors[3], label='y4')
  # pieLegend5 = patches.Patch(color=colors[4], label='y5')
  pieLegend6 = patches.Patch(color=colors[5], label='y6')
  pieLegend7 = patches.Patch(color=colors[6], label='y7')
  pieLegend8 = patches.Patch(color=colors[7], label='y8')
  # pieLegend9 = patches.Patch(color=colors[8], label='y9')
  # pieLegend10 = patches.Patch(color=colors[9], label='y10')
  ax1.legend(handles=[pieLegend2,pieLegend4,pieLegend6,pieLegend7,pieLegend8], loc=(1.02, 0.25), prop={'size': 12})
  
  # show training progress (how many distinct Z fired during each image presentation duration)
  # remove first empty entry
  distinctZFiredHistory.pop(0)
  ax3.plot(distinctZFiredHistory)
  ax3.set_title("Active output neurons", fontsize=14)
  ax3.set_ylabel("# of active output neurons", fontsize=12)
  ax3.set_xlabel("Position [px]", fontsize=12)
  ax3.set_ylim([0.8,2.2])
  ax3.xaxis.set_ticks(np.arange(0,36, 5))
  ax3.yaxis.set_ticks(np.arange(1,3, 1))
  ax3.tick_params(axis="both", labelsize=12)


  for i in range(0, len(ZSpikeHistory[0])):
    ax2.vlines(ZSpikeHistory[1][i], ymin=ZSpikeHistory[0][i] + 1 - 0.5, ymax=ZSpikeHistory[0][i] + 1 + 0.5, color=colors[ZSpikeHistory[0][i]])
  ax2.set_title("Output spikes", fontsize=14)
  ax2.set_ylabel("Output neuron", fontsize=12)
  ax2.set_xlabel("t [s]", fontsize=12)
  ax2.set_ylim([0,11])
  ax2.yaxis.set_ticks(np.arange(1,11, 1))
  ax2.tick_params(axis="both", labelsize=12)
 
  # show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
  ax4.plot(averageZFiredHistory)
  ax4.set_title("Relative activity of most active output neuron", fontsize=14)
  ax4.set_ylabel("Relative activity", fontsize=12)
  ax4.set_xlabel("Position [px]", fontsize=12)
  ax4.xaxis.set_ticks(np.arange(0,36, 5))
  ax4.tick_params(axis="both", labelsize=12)

  
  # plt.tight_layout()
  pickle.dump(fig, open(directoryPath + '/horizontal_validation.pickle','wb'))
  plt.savefig(directoryPath + "/horizontal_validation.svg") 
  plt.show()

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
  # image, position, prior = dataGenerator.generateRandomCrossLineImage()
  image, xPosition, yPosition, prior = dataGenerator.generateCrossLineImageAtPosition(1, 5, 12, imageSize = imageSize, noiseLevel = 0.1, lineThickness = 7)

  fig, ax1 = plt.subplots(1,1)
  # plt.figure()
  ax1.imshow(image, origin='lower', cmap='gray')
  ax1.tick_params(axis="both", labelsize=12)
  ax1.set_title('Input image', fontsize=14)
  fig.savefig(directoryPath + '/crossImage.png')
  images[0].append(image)
  # images[1].append(position)
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
    plt.ylim(0, 10)
  if prior == 0:
    priorString = "vertical"
  else:
    priorString = "horizontal"
  plt.title("Y Spikes for a cross image with prior = '" + priorString + "'", fontsize = 14)
  plt.ylabel("Y Neuron", fontsize=12)
  plt.xlabel("t [s]", fontsize=12)
  plt.tick_params(axis='both', which='major', labelsize=12)
  pickle.dump(fig_object, open(directoryPath + '/crossZSpikes.pickle','wb'))
  plt.savefig(directoryPath + '/crossZSpikes' + str(prior) + '.png')

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

         
