# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:15:48 2023

@author: chris
"""

import dataEncoder
import dataGenerator
import kernel
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import poissonGenerator

plt.close("all")

loadWeights = False
disableIntrinsicWeights = True

# !!! maybe also load parameters of the network together with the weights...?
numberZNeurons = 10
numberANeurons = 1000
numberBNeurons = 3
simulationTime = 500
dt = 0.001
imagePresentationDuration = 0.2
firingRate = 20
imageSize = (29, 29)
numberYNeurons = imageSize[0] * imageSize[1] * 2
sigma = 0.01 # time frame in which spikes count as before output spike
tauRise = 0.001
tauDecay = 0.015
learningRate = 10**-3
Iinh = 0
IinhStartTime = 0
IinhA = 0
IinhAStartTime = 0
tauInh = 0.005

c = 100

# load weights of angle network
angleWeights = np.load("c20_3_YZWeights.npy")
intrinsicAngleWeights = np.zeros(numberZNeurons)

# generate weights of shape network
if loadWeights:
  wZA = np.load("")
  wAA = np.load("")
  WAB = np.load("")
  intrinsicShapeWeights = np.zeros(numberANeurons)
else:
  wZA = np.full((numberBNeurons * numberZNeurons, numberANeurons), -5)
  for angleNetworkIndex in range(numberBNeurons):
    for ZIndex in range(numberZNeurons):
      for thirdDecimalIndex in range(0, 1000, 100):
        for secondDecimalIndex in range(0, 100, 10):
          for firstDecimalIndex in range(0, 10):
            totalZIndex = numberZNeurons * angleNetworkIndex + ZIndex
            totalAIndex = thirdDecimalIndex + secondDecimalIndex + firstDecimalIndex
            if angleNetworkIndex == 0 and ZIndex == firstDecimalIndex:
              wZA[totalZIndex, totalAIndex] = 2
            elif angleNetworkIndex == 1 and ZIndex == secondDecimalIndex / 10:
              wZA[totalZIndex, totalAIndex] = 2
            elif angleNetworkIndex == 2 and ZIndex == thirdDecimalIndex / 100:
              wZA[totalZIndex, totalAIndex] = 2
  wAA = np.full((numberANeurons, numberANeurons - 1), 2)
  intrinsicShapeWeights = np.zeros(numberANeurons)

# define shapes
shapes = []
# triangle
shapes.append([60, 120, 0])
# square
shapes.append([90, 0, 90])
# octagon
shapes.append([135, 90, 45])

##############################################################################
# start simulation
YSpikes = [[[],[]], [[],[]], [[],[]]]
ZSpikes = [[[],[]], [[],[]], [[],[]]]
ASpikes = [[],[]]

for t in np.arange(0, simulationTime, dt):
  # draw random shape every imagePresentationDuration seconds, generate 3 images
  # from the angles and encode them
  if abs(t - round(t / imagePresentationDuration) * imagePresentationDuration) < 1e-10:
    encodedImages = []
    shape = random.choice(shapes)
    for shapeAngle in shape:
      image, angle = dataGenerator.generateImage(shapeAngle, imageSize)
      encodedImages.append(dataEncoder.encodeImage(image))
      
  # every 10ms (i chose this, no special reason, except that Z neuron should spike twice within this time)
  # add random 0-360° to the angles of the chosen shape
  if abs(t - round(t / 0.01) * 0.01) < 1e-10:
    encodedImages = []
    angleOffset = np.random.uniform(0, 360)
    for shapeAngle in shape:
      image, angle = dataGenerator.generateImage(shapeAngle + angleOffset, imageSize)
      encodedImages.append(dataEncoder.encodeImage(image))
  
  ############################################################################
  # generate Z Spikes with angle networks
  # feed angle Networks images that represent a randomly drawn shape
  for networkIterator in range(len(shape)):
    # generate Y Spikes for this step
    for i in range(len(encodedImages[networkIterator])):
      # check if the Yi is active
      if encodedImages[networkIterator][i] == 1:
       # check if Yi spiked in this timestep
       if poissonGenerator.doesNeuronFire(firingRate, dt):
         # when did it spike
         YSpikes[networkIterator][0].append(t)
         # which Y spiked
         YSpikes[networkIterator][1].append(i)
    U = np.zeros(numberZNeurons) + intrinsicAngleWeights
    
    # Next we have to calculate Uk
    expiredYSpikeIDs = []
    YTilde = np.zeros(numberYNeurons)
    for i, YNeuron in enumerate(YSpikes[networkIterator][1]):
      # First mark all YSpikes older than sigma and do not use for calculation of Uk
      if YSpikes[networkIterator][0][i] < t - sigma:
        expiredYSpikeIDs.append(i)
      else:
        YTilde[YSpikes[networkIterator][1][i]] = kernel.YTilde(t, dt, YSpikes[networkIterator][0][i], tauRise, tauDecay)
        for k in range(numberZNeurons):
          U[k] += angleWeights[YNeuron, k] * YTilde[YSpikes[networkIterator][1][i]]
    # delete all spikes that are longer ago than sigma (10ms?) from YSpikes
    for toDeleteID in sorted(expiredYSpikeIDs, reverse=True):
      del YSpikes[networkIterator][0][toDeleteID]
      del YSpikes[networkIterator][1][toDeleteID]
    
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
      ZSpikes[networkIterator][0].append(ZNeuronWinner)
      ZSpikes[networkIterator][1].append(t)
      # calculate inhibition signal and store time of last Z spike
      inhTMP = 0
      for i in range(numberZNeurons):
        inhTMP += np.exp(U[i])
      Iinh = np.log(inhTMP)
      IinhStartTime = t
    elif t - IinhStartTime > tauInh:
      Iinh = 0
      IinhStartTime = math.inf
    
  ############################################################################
  # train shape network
  Ua = np.zeros(numberANeurons) + intrinsicShapeWeights
  # iterate over angle networks
  for networkIterator in range(len(shape)):
  # send all Z Spikes of this angle network to layer A
    expiredZSpikeIDs = []
    ZTilde = np.zeros(numberZNeurons)
    # 
    for i, ZNeuron in enumerate(ZSpikes[networkIterator][0]):
      # First mark all ZSpikes older than sigma and do not use for calculation of Uak
      if ZSpikes[networkIterator][1][i] < t - sigma:
        expiredZSpikeIDs.append(i)
      else:
        ZTilde[ZSpikes[networkIterator][0][i]] = kernel.ZTilde(t, dt, ZSpikes[networkIterator][1][i], tauRise, tauDecay)
        for k in range(numberANeurons):
          # !!!!!! this needs to be adapted (je 3 Z pro A) ... die schleife drüber erstzen durch
          # 3 schleifen von 0 bis 9 (repräsentieren angle networks Z)
          # und wenn ich die drei zähler addiere bekomme ich ID von A
          # mit den drei iteratoren schauen ob die jeweiligen Z gespiked haben
          # und damit das Ua updaten
          Ua[k] += wZA[ZNeuron, k] * ZTilde[ZSpikes[networkIterator][0][i]]
    # delete all spikes that are longer ago than sigma (10ms?) from ZSpikes
    for toDeleteID in sorted(expiredZSpikeIDs, reverse=True):
      del ZSpikes[networkIterator][0][toDeleteID]
      del ZSpikes[networkIterator][1][toDeleteID]
  
  
  
  
  
  
  
  
  
  
  # # calc instantaneous fire rate for each A Neuron for this time step
  # r = np.zeros(numberANeurons)
  # ANeuronsThatWantToFire = []
  # ANeuronWantsToFireAtTime = []
  # # ANeuronFireFactors is used to choose between multiple A firing in this timestep
  # ANeuronFireFactors = []
  # for k in range(numberANeurons):
  #   r[k] = np.exp(Ua[k] - IinhA)
  #   # as far as i understand rk (titled "instantaneous fire rate" in nessler) just says
  #   # how many events occur per second on average
  #   AkFires, ANeuronFireFactor = poissonGenerator.doesANeuronFire(r[k], dt) 
  #   if AkFires:
  #     # mark that Ak wants to fire and also save the time it wants to fire at
  #     ANeuronsThatWantToFire.append(k)
  #     ANeuronWantsToFireAtTime.append(t)
  #     ANeuronFireFactors.append(ANeuronFireFactor)
    
  # # check if any A Neurons want to fire and determine winner A
  # if len(ANeuronsThatWantToFire) > 0:
  #   AFireFactorMax = -math.inf
  #   ANeuronWinner = math.inf
  #   for i in range(len(ANeuronsThatWantToFire)):
  #     if ANeuronFireFactors[i] > AFireFactorMax:  
  #       ANeuronWinner = ANeuronsThatWantToFire[i]
  #       AFireFactorMax = ANeuronFireFactors[i]      
  #   ASpikes[0].append(ANeuronWinner)
  #   ASpikes[1].append(t)
  #   # calculate inhibition signal and store time of last Z spike
  #   inhTMP = 0
  #   for i in range(numberANeurons):
  #     inhTMP += np.exp(Ua[i])
  #   IinhA = np.log(inhTMP)
  #   IinhAStartTime = t
  # elif t - IinhAStartTime > tauInh:
  #   IinhA = 0
  #   IinhAStartTime = math.inf
    
  # if (t) % 1 == 0:
  print("Finished simulation of t= " + str(t))



  
  
  