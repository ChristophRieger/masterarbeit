# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 11:27:49 2022

@author: chris
"""

import math
import numpy as np
import random

# =============================================================================
# returns spike times within the simulationTime
# acc to googling the intervall times of poisson are of exponential distribution
# =============================================================================
def poissonGenerator(firingRate, simulationTime):

  expectedAmountSpikes = firingRate * simulationTime
  spikingTimes = []
  for i in range(expectedAmountSpikes):
    spikingTimes.append(random.expovariate(firingRate))
  spikingTimes = np.cumsum(spikingTimes)
  return spikingTimes

"""
This function generates spiking times according to a poisson process. It draws 
the spiking times from an exponential distribution (acc to googling the 
intervall times of poisson are of exponential distribution) According to our 
experiment the input neurons do not fire between x.4 and x.5 seconds and do 
not fire between x.9 and x.0 seconds.
Parameters:
    firingRate: expected average firing rate of the neurons
    simulationTime: 
returns:
  spike times within the simulationTime and ON-time
"""
def inputPoissonGeneratorOld(firingRate, simulationTime):
  expectedAmountSpikes = firingRate * simulationTime
  # first entry 0 is a mock entry for spikingTimes[-1], this item is removed at end
  spikingTimes = [0]
  absoluteTime = 0
  for i in range(int(expectedAmountSpikes)):
    # time a neuron takes to fire
    relativeSpikingTime = random.expovariate(firingRate)
    # absolute time passed at which this spike happens
    absoluteSpikingTime = relativeSpikingTime + absoluteTime
    absoluteTime += relativeSpikingTime
    spikingTimeSecondDecimalPlace = (absoluteSpikingTime * 10) % 1
    # this controls the off phase of the input neurons
    if spikingTimeSecondDecimalPlace < 0.4 or spikingTimeSecondDecimalPlace > 0.5 and spikingTimeSecondDecimalPlace < 0.9:
      if absoluteSpikingTime < simulationTime:
        spikingTimes.append(absoluteSpikingTime)
  # removing mock entry
  spikingTimes.pop(0)
  return spikingTimes

"""
According to https://www.tu-chemnitz.de/informatik/KI/scripts/ws0910/Neuron_Poisson.pdf
and POBC VO the probability that a neuron fires is firingRate * dt
"""
def doesNeuronFire(firingRate, dt):
  if firingRate * dt > random.random(): 
    return True
  else:
    return False
  
# 2nd parameter is to determine Z winner
def doesZNeuronFire(firingRate, dt):
  randomNumber = random.random()
  if firingRate * dt > randomNumber: 
    return True, firingRate * dt - randomNumber
  else:
    return False, -math.inf
  
def doesANeuronFire(firingRate, dt):
  randomNumber = random.random()
  if firingRate * dt > randomNumber: 
    return True, firingRate * dt - randomNumber
  else:
    return False, -math.inf














