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
import mathematischeAnalyse

# Command Center
loadWeights = False

plt.close("all")

imageSize = (1, 9)
imagePresentationDuration = 20
dt = 0.001 # seconds

firingRate = 200 # Hz; Input neurons yn should spike with 20Hz => firingRate (Lambda) = 20/second
AfiringRate = 400
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

# generate Input data
# no random generation needed for this experiment, rather I use handpicked examples.
# image, prior = dataGenerator.generateRandom1DLineImage(imageSize)
images = [[],[]]
imagesEncoded = []
priorsEncoded = []
PvonYvorausgesetztXundZSimulationList = []
# 1
image = np.array([[0, 255, 0, 255, 255, 255, 0, 255, 0]], dtype=np.uint8)
prior = 0
images[0].append(image)
images[1].append(prior)
# 2
image = np.array([[0, 0, 0, 255, 0, 0, 255, 255, 255]], dtype=np.uint8)
prior = 3
images[0].append(image)
images[1].append(prior)

# 3
image = np.array([[255, 0, 0, 0, 255, 0, 255, 0, 255]], dtype=np.uint8)
prior = 0
images[0].append(image)
images[1].append(prior)
# 4
image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
prior = 3
images[0].append(image)
images[1].append(prior)

# 1 simulation per handpicked input data
for inputIterator in range(len(images[0])):
  XSpikes = [[],[]] # input
  YSpikes = [[],[]] # output
  ZSpikes = [[],[]] # prior
  
  indexOfLastYSpike = [0] * numberXNeurons
  ZNeuronsRecievedYSpikes = [[],[]]
  
  # Metric to measure training progress
  # check how many different Z neurons fired during one image
  distinctZFiredHistory = []
  distinctZFired = []
  averageZFired = []
  averageZFiredHistory = []
  
  # pick current image and prior
  image = images[0][inputIterator]
  prior = images[1][inputIterator]
  
  # start simulation
  for t in np.arange(0, imagePresentationDuration, dt):
    # generate training data every 50ms
    if abs(t - round(t / imagePresentationDuration) * imagePresentationDuration) < 1e-10:
      distinctZFiredHistory.append(len(distinctZFired))
      distinctZFired = []
      if averageZFired:
        mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
        amountMostSpikingZ = averageZFired.count(mostSpikingZ)
        averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
        averageZFired = []
      
    # generate X Spikes for this step
    for i in range(image.shape[1]):
      # check if the Xi is active
      if image[0][i] == 0:
       # check if Xi spiked in this timestep
       if poissonGenerator.doesNeuronFire(firingRate, dt):
         # when did it spike
         XSpikes[0].append(t)
         # which X spiked
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
          U[k] += weights[k, YNeuron] * YTilde[XSpikes[1][i]]
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
      # weights = neuronFunctions.updateWeights(YTilde, weights, ZNeuronWinner, c, learningRate)
      # priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronWinner, c, learningRate)
  
  # Simulation DONE
  
  directoryPath =  "fInput" + str(firingRate) + "_fPrior" + str(AfiringRate) + "_tauDecay" + str(tauDecay)
  if not os.path.exists(directoryPath):
    os.mkdir(directoryPath)
  np.save(directoryPath + "/weights)" + ".npy", weights)
  np.save(directoryPath + "/priorWeights" + ".npy", priorWeights)
    
  
  # 1 hot encode prior
  priorEncoded = np.zeros(4)
  priorEncoded[prior] = 1
  priorsEncoded.append(priorEncoded)
  # 1 hot encode image
  # were image has value 0 => black pixel => should become 1 in encoded image
  imageEncoded = np.zeros(9)
  for i in range(image.shape[1]):
    if image[0, i] == 0:
      imageEncoded[i] = 1
  imagesEncoded.append(imageEncoded)
  
  PvonYvorausgesetztXundZSimulation = np.zeros(4)
  totalSpikes = len(YSpikes[0])
  amountY0Spikes = len(np.where(np.array(YSpikes[0]) == 0)[0])
  amountY1Spikes = len(np.where(np.array(YSpikes[0]) == 1)[0])
  amountY2Spikes = len(np.where(np.array(YSpikes[0]) == 2)[0])
  amountY3Spikes = len(np.where(np.array(YSpikes[0]) == 3)[0])
  PvonYvorausgesetztXundZSimulation[0] = amountY0Spikes / totalSpikes
  PvonYvorausgesetztXundZSimulation[1] = amountY1Spikes / totalSpikes
  PvonYvorausgesetztXundZSimulation[2] = amountY2Spikes / totalSpikes
  PvonYvorausgesetztXundZSimulation[3] = amountY3Spikes / totalSpikes
  PvonYvorausgesetztXundZSimulationList.append(PvonYvorausgesetztXundZSimulation)

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2 * int(len(imagesEncoded)/2), 10 * int(len(imagesEncoded)/2))
counter = 0
for i in range(int(len(imagesEncoded)/2)):
  for j in range(int(len(imagesEncoded)/2)):
    imageEncoded = imagesEncoded[counter]
    image = images[0][counter]
    prior = images[1][counter]
    priorEncoded = priorsEncoded[counter]
    PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulationList[counter]
    PvonYvorausgesetztXundZAnalysis = mathematischeAnalyse.calcPvonYvorausgesetztXundZ(imageEncoded, priorEncoded)  
    
    # gs2 = fig.add_gridspec(6, 2, wspace=0.4, hspace=25)
    # hspace seems to be capped and doesnt really influence the spacing anymore. 
    # but the .svg seems fine anyway, solve only if figure is not good enough.
    # gs3 = fig.add_gridspec(6, 2, wspace=0.4, hspace=400)
    
    ax10 = fig.add_subplot(gs[0 + 2*i, 0 + 2 + 10*j:10 + 10*j - 2])
    
    ax21 = fig.add_subplot(gs[1 + 2*i, 0 + 10*j:4 + 10*j])
    ax22 = fig.add_subplot(gs[1 + 2*i, 6 + 10*j:10 + 10*j])
    
    # Add ghost axes and titles
    ax_firstRow = fig.add_subplot(gs[0 + 2*i, 0 + 10*j:10 + 10*j])
    ax_firstRow.axis('off')
    ax_firstRow.set_title('A', loc="left", x=-0.008,y=0.5, fontsize=16.0, fontweight='semibold')
    
    ax_secondRow = fig.add_subplot(gs[1 + 2*i, 0 + 10*j])
    ax_secondRow.axis('off')
    ax_secondRow.set_title('B', loc="left", x=-0.11,y=0.5, fontsize=16.0, fontweight='semibold')
    
    ax_thirdRow = fig.add_subplot(gs[1 + 2*i, 6 + 10*j])
    ax_thirdRow.axis('off')
    ax_thirdRow.set_title('C', loc="left", x=-0.11,y=0.5, fontsize=16.0, fontweight='semibold')
    
    # plot input data
    ax10.imshow(images[0][counter], cmap='gray')
    rect = patches.Rectangle((-0.5 + prior*2,-0.5), 3, 1, linewidth=8, edgecolor='r', facecolor='none')
    ax10.add_patch(rect)
    rect.set_clip_path(rect)
    ax10.axvline(x=0.5)
    ax10.axvline(x=1.5)
    ax10.axvline(x=2.5)
    ax10.axvline(x=3.5)
    ax10.axvline(x=4.5)
    ax10.axvline(x=5.5)
    ax10.axvline(x=6.5)
    ax10.axvline(x=7.5)
    ax10.axvline(x=8.5)
    ax10.set_ylim([-0.5, 0.5])
    ax10.set_title("Input data", y=1.3)
    ax10.axes.yaxis.set_visible(False)
    
    
    PvonYvorausgesetztXundZAnalysis = PvonYvorausgesetztXundZAnalysis.reshape(4,1)
    tab21 = ax21.table(cellText=np.around(PvonYvorausgesetztXundZAnalysis, 3), loc='center')
    tab21.auto_set_font_size(False)
    tab21.auto_set_column_width(0)
    tab21.scale(1,1.2)
    ax21.axis('off')
    ax21.set_title("Analysis output probabilities", y=1.1)
    
    PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulation.reshape(4,1)
    tab22 = ax22.table(cellText=np.around(PvonYvorausgesetztXundZSimulation, 3), loc='center')
    tab22.auto_set_font_size(False)
    tab22.auto_set_column_width(0)
    tab22.scale(1,1.2)
    ax22.axis('off')
    ax22.set_title("Simulation output probabilities", y=1.1)
    
    counter += 1
  
pickle.dump(fig, open(directoryPath + "/trainingPlot2" + '.pickle','wb'))
plt.savefig(directoryPath + "/trainingPlot2" + ".svg")  
plt.savefig(directoryPath + "/trainingPlot2" + ".png")
plt.savefig(directoryPath + "/trainingPlot2" + ".jpg") 
plt.show()

