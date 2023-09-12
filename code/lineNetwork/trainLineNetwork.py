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
simulationTime = 10
dt = 0.001 # seconds

# 100/500 and 0.003 seems nice to me
firingRate = 98 # Hz;
AfiringRate = 440
numberXNeurons = imageSize[0] * imageSize[1] * 2# 1 neurons per pixel (one for black) # input
numberYNeurons = 4 # output
numberZNeurons = 4 # prior

sigma = 0.01 # time frame in which spikes count as before output spike
c = 20 # 10 seems good, scales weights to 0 ... 1
tauRise = 0.001
tauDecay = 0.004  # might be onto something here, my problem gets better

learningRateFactor = 3
learningRate = 10**-learningRateFactor
RStar = 200 # Hz; total output firing rate

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
for t in np.arange(0, imagePresentationDuration, dt):
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
    weights = neuronFunctions.updateWeights(XTilde, weights, ZNeuronsThatWantToFire[i], c, learningRate)
    priorWeights = neuronFunctions.updateWeights(ATilde, priorWeights, ZNeuronsThatWantToFire[i], c, learningRate)
    
# Simulation DONE ############
    
directoryPath =  "fInput" + str(firingRate) + "_fPrior" + str(AfiringRate) + "_tauDecay" + str(tauDecay)
if not os.path.exists(directoryPath):
  os.mkdir(directoryPath)
np.save(directoryPath + "/weights" + ".npy", weights)
np.save(directoryPath + "/priorWeights" + ".npy", priorWeights)

# TRAINING PLOT
colors = ['red', 'blue', 'black', 'green']
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

pickle.dump(fig, open(directoryPath + "/trainingPlot" + '.pickle','wb'))
plt.savefig(directoryPath + "/trainingPlot.svg")  
plt.savefig(directoryPath + "/trainingPlot.png")
plt.savefig(directoryPath + "/trainingPlot.jpg") 
plt.show()




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
#     ax_firstRow.set_title('A' + str(counter+1), loc="left", x=-0.008,y=0.5, fontsize=16.0, fontweight='semibold')
    
#     ax_secondRow = fig.add_subplot(gs[1 + 2*i, 0 + 10*j])
#     ax_secondRow.axis('off')
#     ax_secondRow.set_title('B' + str(counter+1), loc="left", x=-0.11,y=0.5, fontsize=16.0, fontweight='semibold')
    
#     ax_thirdRow = fig.add_subplot(gs[1 + 2*i, 6 + 10*j])
#     ax_thirdRow.axis('off')
#     ax_thirdRow.set_title('C' + str(counter+1), loc="left", x=-0.11,y=0.5, fontsize=16.0, fontweight='semibold')
    
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