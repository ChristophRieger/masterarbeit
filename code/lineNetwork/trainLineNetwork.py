# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:46:32 2023

@author: chris
"""
import sys
sys.path.insert(0, '../helpers')
import dataGenerator
import poissonGenerator
import neuronFunctions
import kernel
import dataEncoder

import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import os
import pickle
import mathematischeAnalyse
import copy

# Command Center
loadWeights = False

plt.close("all")

imageSize = (1, 9)
imagePresentationDuration = 0.2
simulationTime = 800
dt = 0.001 # seconds

# 100/500 and 0.003 seems nice to me
firingRate = 98 # Hz;
AfiringRate = 440
numberXNeurons = imageSize[0] * imageSize[1] * 2# 1 neurons per pixel (one for black) # input
numberYNeurons = 4 # output
numberZNeurons = 4 # prior

sigma = 0.01 # time frame in which spikes count as before output spike
c = 2.5 # 10 seems good, scales weights to 0 ... 1
tauRise = 0.001
tauDecay = 0.004  # might be onto something here, my problem gets better

learningRateFactor = 3
learningRate = 10**-learningRateFactor
RStar = 200 # Hz; total output firing rate

analyticWeights = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]],
                      "float64")
# analyticWeights = np.log(analyticWeights)

analyticPriorWeights = np.array([[0.9, 0.0333, 0.0333, 0.0333],
                         [0.0333, 0.9, 0.0333, 0.0333],
                         [0.0333, 0.0333, 0.9, 0.0333],
                         [0.0333, 0.0333, 0.0333, 0.9]],
                      "float64")
# analyticPriorWeights = np.log(analyticPriorWeights)

# initialize weights (for now between 0 and 1, not sure)
if loadWeights:
  weights = np.load("c20_3_YZWeights.npy")
  priorWeights = np.load("c20_3_AZWeights.npy")
else:
  weights = np.full((numberYNeurons, numberXNeurons), 2, "float64")
  priorWeights = np.full((numberYNeurons, numberZNeurons), 2, "float64")
  
XSpikes = [[],[]] # input
YSpikes = [[],[]] # output
ZSpikes = [[],[]] # prior
images = [[],[]]
#       indexOfLastYSpike = [0] * numberXNeurons
#       ZNeuronsRecievedYSpikes = [[],[]]
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
    image, prior = dataGenerator.generateRandom1DLineImage()
    images[0].append(image)
    images[1].append(prior)
    encodedImage = dataEncoder.encodeImage(image)
    
    distinctZFiredHistory.append(len(distinctZFired))
    distinctZFired = []
    if averageZFired:
      mostSpikingZ = max(set(averageZFired), key = averageZFired.count)
      amountMostSpikingZ = averageZFired.count(mostSpikingZ)
      averageZFiredHistory.append(amountMostSpikingZ / len(averageZFired))
      averageZFired = []
  
  # generate X Spikes for this step
  for i in range(len(encodedImage)):
    # check if the Xi is active
    if encodedImage[i] == 0:
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
  # Add contribution of X
  expiredYSpikeIDs = []
  XTilde = np.zeros(numberXNeurons)
  for i, XNeuron in enumerate(XSpikes[1]):
    # First mark all XSpikes older than sigma and do not use for calculation of Uk
    if XSpikes[0][i] < t - sigma:
      expiredYSpikeIDs.append(i)
    else:
      XTilde[XSpikes[1][i]] = kernel.tilde(t, dt, XSpikes[0][i], tauRise, tauDecay)
      for k in range(numberYNeurons):
        U[k] += weights[k, XNeuron] * XTilde[XSpikes[1][i]]
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
  
  # let all output neurons fire that want to fire
  for i in range(len(ZNeuronsThatWantToFire)):
    YSpikes[0].append(ZNeuronsThatWantToFire[i])
    YSpikes[1].append(t)
    averageZFired.append(ZNeuronsThatWantToFire[i])  
    # append ID of Z if this Z has not fired yet in this imagePresentationDuration
    if not distinctZFired.count(ZNeuronsThatWantToFire[i]):
      distinctZFired.append(ZNeuronsThatWantToFire[i]) 
    weights = neuronFunctions.updateWeights(XTilde, weights.T, ZNeuronsThatWantToFire[i], c, learningRate)
    # transpose weights again because everything is fucked up
    weights = weights.T
    priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronsThatWantToFire[i], c, learningRate)
  if (t) % 1 == 0:
      print("Finished simulation of t= " + str(t))
# Simulation DONE ############
    
directoryPath =  "training_" + str(firingRate) + "_" + str(AfiringRate) + "_" + str(tauDecay) + "_" + str(c)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)

# sort the weights so they align with the analytic weights
weightsTmp = np.full((numberYNeurons, numberXNeurons), 0, "float64")
for j in range(4):
  maxWeight = -99999
  rowToMove = 99999
  for i in range(4):
    if weights[i,2 + j * 4] > maxWeight:
      maxWeight = weights[i,2 + j * 4]
      rowToMove = i
  weightsTmp[j, :] = weights[rowToMove, :]  
weights = weightsTmp

priorWeightsTmp = np.full((numberYNeurons, numberZNeurons), 0, "float64")
for j in range(4):
  maxWeight = -99999
  rowToMove = 99999
  for i in range(4):
    if priorWeights[i, j] > maxWeight:
      maxWeight = priorWeights[i, j]
      rowToMove = i
  priorWeightsTmp[j, :] = priorWeights[rowToMove, :]  
priorWeights = priorWeightsTmp

np.save(directoryPath + "/weights" + ".npy", weights)
np.save(directoryPath + "/priorWeights" + ".npy", priorWeights)

# TRAINING PLOT
colors = ['red', 'blue', 'black', 'green']
fig = plt.figure(figsize=(12,16))
gs = fig.add_gridspec(8, 2, hspace=0.4)
gs2 = fig.add_gridspec(8, 2, wspace=0.3, hspace=1.4)
# hspace seems to be capped and doesnt really influence the spacing anymore. 
# but the .svg seems fine anyway, solve only if figure is not good enough.
gs3 = fig.add_gridspec(8, 2, wspace=0.3, hspace=1.4)


ax10 = fig.add_subplot(gs[0:1, 0])
ax11 = fig.add_subplot(gs[0:1, 1])
ax20 = fig.add_subplot(gs[1:2, 0])
ax21 = fig.add_subplot(gs[1:2, 1])

ax31 = fig.add_subplot(gs2[2:5, 0])
ax32 = fig.add_subplot(gs2[2:5, 1])

# ax41 = fig.add_subplot(gs3[5:8, 0])
ax42 = fig.add_subplot(gs3[5:8, 0:2])

# Add ghost axes and titles
ax_firstRowFirstColumn = fig.add_subplot(gs[0:2, 0])
ax_firstRowFirstColumn.axis('off')
ax_firstRowFirstColumn.set_title('A', loc="left", x=-0.06,y=1, fontsize=16.0, fontweight='semibold')
ax_firstRowSecondColumn = fig.add_subplot(gs[0:2, 1])
ax_firstRowSecondColumn.axis('off')
ax_firstRowSecondColumn.set_title('B', loc="left", x=-0.06,y=1, fontsize=16.0, fontweight='semibold')
ax_secondRowFirstColumn = fig.add_subplot(gs[2:4, 0])
ax_secondRowFirstColumn.axis('off')
ax_secondRowFirstColumn.set_title('C', loc="left", x=-0.06,y=1.5, fontsize=16.0, fontweight='semibold')
ax_secondRowSecondColumn = fig.add_subplot(gs[2:4, 1])
ax_secondRowSecondColumn.axis('off')
ax_secondRowSecondColumn.set_title('D', loc="left", x=-0.06,y=1.5, fontsize=16.0, fontweight='semibold')

ax_thirdRowFirstColumn = fig.add_subplot(gs2[4:6, 0])
ax_thirdRowFirstColumn.axis('off')
ax_thirdRowFirstColumn.set_title('E', loc="left", x=-0.06, y=2.425, fontsize=16.0, fontweight='semibold')
ax_thirdRowSecondColumn = fig.add_subplot(gs2[4:6, 1])
ax_thirdRowSecondColumn.axis('off')
# ax_thirdRowSecondColumn.set_title('F', loc="left", x=-0.151, y=2.425, fontsize=16.0, fontweight='semibold')
ax_thirdRowSecondColumn.set_title('F', loc="left", x=-0.06, y=2.425, fontsize=16.0, fontweight='semibold')

ax_fourthRowFirstColumn = fig.add_subplot(gs3[6:8, 0:2])
ax_fourthRowFirstColumn.axis('off')
ax_fourthRowFirstColumn.set_title('G', loc="left", x=-0.06, y=1.715, fontsize=16.0, fontweight='semibold')
# ax_fourthRowSecondColumn = fig.add_subplot(gs3[6:8, 1])
# ax_fourthRowSecondColumn.axis('off')
# ax_fourthRowSecondColumn.set_title('H', loc="left", x=-0.151, y=1.715, fontsize=16.0, fontweight='semibold')


# plot weights
tab10 = ax10.table(cellText=np.around(np.exp(weights[:, 0::2]), 2), loc='center', cellLoc='center'
, colWidths=[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
tab10.scale(1,1.8)
# tab10.auto_set_column_width([0,1,2,3,4,5,6,7,8])
tab10.auto_set_font_size(False)
tab10.set_fontsize(13)
ax10.axis('off')
ax10.set_title("Learned $P^{X|Y}$", y=1)
ax10.title.set_size(14)
tab11 = ax11.table(cellText=np.around(analyticWeights, 2), loc='center', cellLoc='center'
, colWidths=[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
tab11.scale(1,1.8)
# tab11.auto_set_column_width([0,1,2,3,4,5,6,7,8])
tab11.auto_set_font_size(False)
tab11.set_fontsize(13)
ax11.axis('off')
ax11.set_title("Calculated $P^{X|Y}$", y=1)
ax11.title.set_size(14)
tab20 = ax20.table(cellText=np.around(np.exp(priorWeights), 2), loc='center', cellLoc='center'
, colWidths=[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
tab20.scale(1,1.8)
# tab20.auto_set_column_width([0,1,2,3,4,5,6,7,8])
tab20.auto_set_font_size(False)
tab20.set_fontsize(13)
ax20.axis('off')
ax20.set_title("Learned $P^{Y|Z}$", y=1)
ax20.title.set_size(14)
tab21 = ax21.table(cellText=np.around(analyticPriorWeights, 2), loc='center', cellLoc='center'
, colWidths=[0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11])
tab21.scale(1,1.8)
# tab21.auto_set_column_width([0,1,2,3,4,5,6,7,8])
tab21.auto_set_font_size(False)
tab21.set_fontsize(13)
ax21.axis('off')
ax21.set_title("Calculated $P^{Y|Z}$", y=1)
ax21.title.set_size(14)

# plot the first 120 output spikes
for i in range(0, 120):
  ax31.vlines(YSpikes[1][i], ymin=YSpikes[0][i] + 1 - 0.5, ymax=YSpikes[0][i] + 1 + 0.5, color=colors[YSpikes[0][i]], linewidth=0.5)
ax31.axvline(x=0, color="red", linestyle="dashed")
ax31.axvline(x=0.2, color="red", linestyle="dashed")
ax31.axvline(x=0.4, color="red", linestyle="dashed")
ax31.axvline(x=0.6, color="red", linestyle="dashed")
ax31.set_title("Output neuron activity before learning")
ax31.title.set_size(14)
ax31.set_ylabel("Output neuron", fontsize=12)
ax31.set_xlabel("Time [s]", fontsize=12)
ax31.tick_params(axis='both', labelsize=11)
ax31.yaxis.set_ticks(np.arange(1,5, 1))

# plot the last 120 Z spikes
for i in range(len(YSpikes[0]) - len(YSpikes[0][-120:]), len(YSpikes[0])):
  ax32.vlines(YSpikes[1][i], ymin=YSpikes[0][i] + 1 - 0.5, ymax=YSpikes[0][i] + 1 + 0.5, color=colors[YSpikes[0][i]], linewidth=0.5)
ax32.axvline(x=simulationTime, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.2, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.4, color="red", linestyle="dashed")
ax32.axvline(x=simulationTime - 0.6, color="red", linestyle="dashed")
ax32.set_title("Output neuron activity after learning")
ax32.title.set_size(14)
ax32.set_ylabel("Output neuron", fontsize=12)
ax32.set_xlabel("Time [s]", fontsize=12)
ax32.tick_params(axis='both', labelsize=11)
ax32.yaxis.set_ticks(np.arange(1,5, 1))


# show training progress (how many distinct Z fired during each image presentation duration)
# remove first empty entry
# distinctZFiredHistory.pop(0)
# ax41.plot(distinctZFiredHistory)
# ax41.set_title("Training progress")
# ax41.title.set_size(14)
# ax41.set_ylabel("# of output neurons spiking", fontsize=12)
# ax41.set_xlabel("Image shown", fontsize=12)
# ax41.tick_params(axis='both', labelsize=11)

# show training progress (fraction of spikes of most common Z neuron to amount of overall Z spikes)
ax42.plot(averageZFiredHistory)
ax42.set_title("Relative activity of most active output neuron")
ax42.title.set_size(14)
ax42.set_ylabel("Relative activity", fontsize=12)
ax42.set_xlabel("Image shown", fontsize=12)
ax42.tick_params(axis='both', labelsize=11)

pickle.dump(fig, open(directoryPath + "/trainingPlot" + '.pickle','wb'))
plt.savefig(directoryPath + "/trainingPlot.svg", bbox_inches='tight')  
plt.savefig(directoryPath + "/trainingPlot.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()



# # 1 hot encode prior
# priorEncoded = np.zeros(4)
# # TODO  !!!!!!!! disabled next line to remove prior 
# priorEncoded[prior] = 1
# priorsEncoded.append(priorEncoded)
# # 1 hot encode image
# # were image has value 0 => black pixel => should become 1 in encoded image
# imageEncoded = np.zeros(9)
# for i in range(image.shape[1]):
#   if image[0, i] == 0:
#     imageEncoded[i] = 1
# imagesEncoded.append(imageEncoded)

# PvonYvorausgesetztXundZSimulation = np.zeros(4)
# totalSpikes = len(YSpikes[0])
# amountY0Spikes = len(np.where(np.array(YSpikes[0]) == 0)[0])
# amountY1Spikes = len(np.where(np.array(YSpikes[0]) == 1)[0])
# amountY2Spikes = len(np.where(np.array(YSpikes[0]) == 2)[0])
# amountY3Spikes = len(np.where(np.array(YSpikes[0]) == 3)[0])
# PvonYvorausgesetztXundZSimulation[0] = amountY0Spikes / totalSpikes
# PvonYvorausgesetztXundZSimulation[1] = amountY1Spikes / totalSpikes
# PvonYvorausgesetztXundZSimulation[2] = amountY2Spikes / totalSpikes
# PvonYvorausgesetztXundZSimulation[3] = amountY3Spikes / totalSpikes
# PvonYvorausgesetztXundZSimulationList.append(PvonYvorausgesetztXundZSimulation)
# PvonYvorausgesetztXundZSimulationListList.append(PvonYvorausgesetztXundZSimulationList)
# fig = plt.figure(figsize=(20, 12))
# gs = fig.add_gridspec(2 * int(len(imagesEncoded)/2) + 1, 20)
# counter = 0
# klDivergenceList = [[],[],[],[],[],[]]
# for i in range(int(len(imagesEncoded)/2)):
#   for j in range(2):
#     imageEncoded = imagesEncoded[counter]
#     image = images[0][counter]
#     prior = images[1][counter]
#     priorEncoded = priorsEncoded[counter]
    
#     # calc std and average of the 20 runs
#     PvonYvorausgesetztXundZSimulationMeanTmp = np.zeros(4)
#     PvonYvorausgesetztXundZSimulationStdTmp = [[],[],[],[]]
#     for runsIterator in range(20):
#       for outputClassIterator in range(numberYNeurons):
#         PvonYvorausgesetztXundZSimulationMeanTmp[outputClassIterator] += PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator]
#         PvonYvorausgesetztXundZSimulationStdTmp[outputClassIterator].append(PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator])
    
#     # calc mean of simulation probabs over 20 runs
#     PvonYvorausgesetztXundZSimulationMeanTmp /= 20
#     PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulationMeanTmp
    
#     # calc standard deviation of simulation probabs over 20 runs
#     standardDeviations = np.zeros(numberYNeurons)
#     for outputClassIterator in range(numberYNeurons):
#       standardDeviations[outputClassIterator] = np.std(PvonYvorausgesetztXundZSimulationStdTmp[outputClassIterator])
      
    
#     PvonYvorausgesetztXundZAnalysis = mathematischeAnalyse.calcPvonYvorausgesetztXundZ(imageEncoded, priorEncoded)
#     # TODO changed function to disable prior
#     # PvonYvorausgesetztXundZAnalysis = mathematischeAnalyse.calcPvonYvorausgesetztXundZNull(imageEncoded, priorEncoded)
    
#     # gs2 = fig.add_gridspec(6, 2, wspace=0.4, hspace=25)
#     # hspace seems to be capped and doesnt really influence the spacing anymore. 
#     # but the .svg seems fine anyway, solve only if figure is not good enough.
#     # gs3 = fig.add_gridspec(6, 2, wspace=0.4, hspace=400)
    
#     ax10 = fig.add_subplot(gs[0 + 2*i, 0 + 2 + 10*j:10 + 10*j - 2])
    
#     ax21 = fig.add_subplot(gs[1 + 2*i, 0 + 10*j:4 + 10*j])
#     ax22 = fig.add_subplot(gs[1 + 2*i, 6 + 10*j:10 + 10*j])
    
#     # Add ghost axes and titles
#     ax_firstRow = fig.add_subplot(gs[0 + 2*i, 0 + 10*j:10 + 10*j])
#     ax_firstRow.axis('off')
#     ax_firstRow.set_title('A' + str(counter+1), loc="left", x=-0.008,y=0.5, fontsize=18.0, fontweight='semibold')
    
#     ax_secondRow = fig.add_subplot(gs[1 + 2*i, 0 + 10*j])
#     ax_secondRow.axis('off')
#     ax_secondRow.set_title('B' + str(counter+1), loc="left", x=-0.11,y=0.5, fontsize=18.0, fontweight='semibold')
    
#     ax_thirdRow = fig.add_subplot(gs[1 + 2*i, 6 + 10*j])
#     ax_thirdRow.axis('off')
#     ax_thirdRow.set_title('C' + str(counter+1), loc="left", x=-0.11,y=0.5, fontsize=18.0, fontweight='semibold')
    
#     # plot input data
#     ax10.imshow(images[0][counter], cmap='gray')
#     # TODO !!!!!! remove next 3 lines to disable prior
#     rect = patches.Rectangle((-0.5 + prior*2,-0.5), 3, 1, linewidth=8, edgecolor='r', facecolor='none')
#     ax10.add_patch(rect)
#     rect.set_clip_path(rect)
#     ax10.axvline(x=0.5)
#     ax10.axvline(x=1.5)
#     ax10.axvline(x=2.5)
#     ax10.axvline(x=3.5)
#     ax10.axvline(x=4.5)
#     ax10.axvline(x=5.5)
#     ax10.axvline(x=6.5)
#     ax10.axvline(x=7.5)
#     ax10.axvline(x=8.5)
#     ax10.set_ylim([-0.5, 0.5])
#     ax10.set_title("Input data", y=1.3)
#     ax10.axes.yaxis.set_visible(False)
    
    
#     PvonYvorausgesetztXundZAnalysis = PvonYvorausgesetztXundZAnalysis.reshape(4,1)
#     tab21 = ax21.table(cellText=np.around(PvonYvorausgesetztXundZAnalysis, 3), loc='center')
#     tab21.auto_set_font_size(False)
#     tab21.auto_set_column_width(0)
#     tab21.scale(1,1.2)
#     ax21.axis('off')
#     ax21.set_title("Analysis output \n probabilities", y=0.9)
    
#     PvonYvorausgesetztXundZSimulation = PvonYvorausgesetztXundZSimulation.reshape(4,1)
#     standardDeviations = standardDeviations.reshape(4,1)
#     cellTextTmp = [[], [], [], []]
#     for cellTextIterator in range(numberYNeurons):  
#       cellTextTmp[cellTextIterator].append(str(np.around(PvonYvorausgesetztXundZSimulation[cellTextIterator], 3)).strip("[]") + " (" + str(np.around(standardDeviations[cellTextIterator], 4)).strip("[]") + ")")
#     tab22 = ax22.table(cellText=cellTextTmp, loc='center')
#     tab22.auto_set_font_size(False)
#     tab22.auto_set_column_width(0)
#     tab22.scale(1,1.2)
#     ax22.axis('off')
#     ax22.set_title("Simulation output \n probabilities", y=0.9)
    
#     # calculate Kullback Leibler Divergence for each image and each run
#     for runsIterator in range(20):
#       klDivergenceTmp = 0
#       for outputClassIterator in range(numberYNeurons):
#         klDivergenceTmp += PvonYvorausgesetztXundZAnalysis[outputClassIterator] * np.log(PvonYvorausgesetztXundZAnalysis[outputClassIterator] / PvonYvorausgesetztXundZSimulationListList[runsIterator][counter][outputClassIterator]) 
#       klDivergenceList[counter].append(klDivergenceTmp)
    
#     counter += 1

# klDivergenceMeanPerRun = []
# for runIterator in range(20):
#   klDivergenceMeanPerRunTmp = 0
#   for imageIterator in range(6):
#     klDivergenceMeanPerRunTmp += klDivergenceList[imageIterator][runIterator]
#   klDivergenceMeanPerRun.append(klDivergenceMeanPerRunTmp / 6)
   
    
# klDivergenceMean = sum(klDivergenceMeanPerRun) / len(klDivergenceMeanPerRun)
# klDivergenceStd = np.std(klDivergenceMeanPerRun)
# ax3 = fig.add_subplot(gs[2 * int(len(imagesEncoded)/2):2 * int(len(imagesEncoded)/2)+1, 0:20])
# ax3.axis('off')
# textStyle = dict(horizontalalignment='center', verticalalignment='center',
#                 fontsize=16)
# ax3.text(0.5, 0.5, "Kullback–Leibler divergence = " + str(np.around(klDivergenceMean, 4)).strip("[]") + " (" + str(np.around(klDivergenceStd, 5)) + ")", textStyle, transform=ax3.transAxes)

# pickle.dump(fig, open(directoryPath + "/trainingPlot5" + '.pickle','wb'))
# plt.savefig(directoryPath + "/trainingPlot5" + ".svg")  
# plt.savefig(directoryPath + "/trainingPlot5" + ".png")
# plt.savefig(directoryPath + "/trainingPlot5" + ".jpg") 
# plt.show()