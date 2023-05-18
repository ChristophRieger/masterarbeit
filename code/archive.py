# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:20:38 2023

@author: chris
"""



###### logic to generate exact prior neuron frequency spikes, not needed anymore
# because i have more than 2 prior neurons and f is also higher than before

# generate A Spikes for this step
      # generate for priorNeuron0 Z0
      if poissonGenerator.doesNeuronFire(currentPrior, dt):
        ASpikes[0].append(t)
        ASpikes[1].append(0)
        ASpikeHistory.append(0)
      # generate for priorNeuron1 Z1
      if poissonGenerator.doesNeuronFire(AfiringRate - currentPrior, dt):
        ASpikes[0].append(t)
        ASpikes[1].append(1)         
        ASpikeHistory.append(1)
    # check firing frequency of both A neurons
    # gets ids where A0 spiked
    # remove A spikes if we have too many
    if ASpikes[1].count(0) > currentPrior:
      # loop for how many we have to remove
      for removeCounter in range(0, ASpikes[1].count(0) - currentPrior):
        spikeIds = [i for i,x in enumerate(ASpikes[1]) if x == 0]
        chosenIdToRemove = random.choice(spikeIds)
        spikeIds.remove(chosenIdToRemove)
        del ASpikes[0][chosenIdToRemove]
        del ASpikes[1][chosenIdToRemove]
        del ASpikeHistory[chosenIdToRemove]
    # add ASpikes if we have to few
    elif ASpikes[1].count(0) < currentPrior:
      # loop for how many we have to add
      for addCounter in range(0, currentPrior - ASpikes[1].count(0)):
        addedASpike = False
        while not addedASpike:
          spikeIds = [i for i,x in enumerate(ASpikes[1]) if x == 0]
          # generate random time in 1ms steps
          randomTime = math.ceil(random.random() * 1000)/1000
          # check if A already spikes at random time, if it does, move to next iteration
          res_list = [ASpikes[0][spikeId] for spikeId in spikeIds]
          if res_list.count(randomTime) != 0:
            continue
          # add a spike at the random time
          ASpikes[0].append(randomTime)
          ASpikes[1].append(0)
          ASpikeHistory.append(0)
          spikeIds.append(len(spikeIds))
          addedASpike = True
          
    # gets ids where A1 spiked
    # remove A spikes if we have too many
    if ASpikes[1].count(1) > AfiringRate - currentPrior:
      # loop for how many we have to remove
      for removeCounter in range(0, ASpikes[1].count(1) - (AfiringRate - currentPrior)):
        spikeIds = [i for i,x in enumerate(ASpikes[1]) if x == 1]
        chosenIdToRemove = random.choice(spikeIds)
        spikeIds.remove(chosenIdToRemove)
        del ASpikes[0][chosenIdToRemove]
        del ASpikes[1][chosenIdToRemove]
        del ASpikeHistory[chosenIdToRemove]
    # add ASpikes if we have to few
    elif ASpikes[1].count(1) < AfiringRate - currentPrior:
      # loop for how many we have to add
      for addCounter in range(0, AfiringRate - currentPrior - ASpikes[1].count(1)):
        addedASpike = False
        while not addedASpike:
          spikeIds = [i for i,x in enumerate(ASpikes[1]) if x == 1]
          # generate random time in 1ms steps
          randomTime = math.ceil(random.random() * 1000)/1000
          # check if A already spikes at random time, if it does, move to next iteration
          res_list = [ASpikes[0][spikeId] for spikeId in spikeIds]
          if res_list.count(randomTime) != 0:
            continue
          # add a spike at the random time
          ASpikes[0].append(randomTime)
          ASpikes[1].append(1)
          ASpikeHistory.append(1)
          spikeIds.append(len(spikeIds))
          addedASpike = True