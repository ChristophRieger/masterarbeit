## 1 Hallo

Good day and welcome to the presentation of my master thesis with the title hierarchical architectures for spiking Winner-Take-All networks. It was supervised by professor Legenstein.

## 3 Biological Background 1

Spiking neural networks are artificial neural networks that resemble biological neural networks closely. Neurons in typical artificial neural networks transmit information at every propagation cycle. This, however, is not how biological neurons operate. They generate spikes when their membrane potentials exceed a threshold.
The membrane potentials decay over time, and this is modeled via a kernel function. The kernel function gives the basic amplitude of a spike over time. As this curve falls the membrane potential also decreases again.

The used model here is also a winner-take-all network. This means, that only 1 neuron at a time can produce some output. This mechanism was also observed in biology for pyramidal neurons. These neurons send signals from one network to another, while inhibiting neighbouring pyramidal neurons.

The brain is thought to utilize probabilistic computing to solve tasks. Measurements of the brain show, that neural activity is different every time, even when the same task is performed. This suggests that the averaged neural activity of neurons matters. Because of that this thesis treats the network model in a probabilistic way.

Neurons in the brain are connected to each other via synapses. It was shown that the postsynaptic response amplitude changes over time. Depending on the stimulation the response gets larger or smaller. This is how the brain learns and forms memories. In network models this effect is modelled by synaptic weights.
The learning rule used in this model is spike timing dependent plasticity. This learning rule looks at the timing of pre and postsynaptic spikes. When a presynaptic spike happens shortly before a postsynaptic spike, then the spike helped to generate the postsynaptic spike. Thus the synaptic weight is increased. When no presynaptic spike occured, then the weight is decreased.



## 4 Biological Background 2

Biological neural networks are organized in a hierarchical structure. This means, that the concepts a network represents get more complicated as you go up the hierarchy. For example V1 of the visual cortex represents horizontal and vertical bars at specific points in your visual field. The inferior temporal cortex which is high up in the hierarchy has neurons that specifically activiate when you look at a face.

Most of the information flows from the bottom of the hierarchy to the top. But there also exists feedback, that is passed from the top back to the bottom. This is needed for attention and biased competition. Lee and Mumford performed experiments with monkeys, where they showed that feedback can also cause neurons to see illusions. For this they made monkeys look at Kanizsa squares, while measuring neuronal activity in V1 and V2 of the visual cortex. When you look at this image, you are supposed to see a grey square lying on top of the black disks. The neurons of V1 they measured had very small receptive fields and would activiate when there is a horizontal edge in its field. Usually they would activate after 45ms. When they were measured while looking at the top middle of the illusory square they would not activate at first. However, after 100 ms they activated, indicated that they were seeing a horizontal edge. They claimed that this was proof of feedback from a network higher up in the hierarchy which had more context about the whole image.

## 5 Theoretical Background 

This model is based on Bayesian Inference. Bayesian Inference gives the probability of an hypothesis, given related evidence.

When applying the experiment of the illusory contour to Bayesian Inference we get the following formula. 
Here X is a random binary vector of an input image, 
Y is a multinomial variable of the output of V1
and Z is a multinomial variable of the feedback from some network higher up in the hierarchy. 
At the left side of the equation we have the bayesian posterior with P of Y given X and Z. P of X given Y is called the likelihood. And P of Y given Z is called the prior.
It is hypothesized that these Bayesian probabilities are all represented by the activity of the network. The output of the network is supposed to represent the posterior, the activity of the input neurons is the likelihood and the activity of the feedback neurons are the prior. This will be analysed in the experiments.

The network model used in this thesis was taken from Nessler et al. and expanded by an additional layer of prior neurons which deliver feedback. In the thesis I mathematically proved, that this expansion is valid and everything still makes sense.

Nessler et al. also claimed that the synaptic input weights of the model converge towards the log of the Bayesian likelihood. This will also be analysed in the experiments.

## 6 Network image

On the right we see a visualization of the network.
Input image black white. The input neurons are given as x, and every pixel of the image is connected to two input neurons, one active for white pixels, one active for black pixels.
Input neurons firing with constant frequency when active and are fully connected to the output neurons y.
Output neurons each have a membrane potential and they fire according to poisson process. Their firing rate is given by e to the power of the membrane potential minus inhibition
Prior neurons are the expansion to the nessler model. They also fire constantly when active, fully connected, prior weights

To achieve the winner take all behaviour of the output neurons an adaptive inhibition function was used. This inhibition takes 2 parameters. The first one is the log of the target spiking rate for all the output neurons combined. It was 200 Hertz. The second parameter is the log of the sum of all firing rates of the output neurons. This inhibition signal was then subtracted from each membrane potential of the output neurons, realizing a winner take all behaviour.

On the left we have the most important equations of the model.
The membrane potential uk of t of an output neuron yk  is given by the sum of two sums. The first sum over all input neurons of the input weight times the unweighed membrane response xi of t. Xi is obtained by summing the amplitudes of all spikes that a input neuron emitted.
The second part of the membrane potential is a sum similar to the first part, but this time for the prior neurons. This second sum is my extension of the prior neurons.

The probability that an output neurons generates a spike at time t is given by the second equation. This probability is proportional to e to the power of the membrane potential minus the inhibition.

The conditional probability that an output spike generated from output neuron k is given by qk of t. This is essentially the prediction the model performs. It is given by the firing rate of an output neuron, divided by the sum of all firing rates.
The firing rates are given by e to the power of the membrane potential minus the inhibition. However, as the inhibition does not depend on which output neuron we look at, it cancels out.

And finally the input weights are 


## 9 Methodology

To evaluate the performance of the network the Kullback-Leibler divergence was chosen. It is a statistical distance that gives how much information is lost when one probability distribution is used to approximate another.
To obtain the Kullback-Leibler divergence the firing rates of the output neurons were compared to the Bayesian posterior.

## 10 Ambiguous visual image 2

There were 20 prior neurons. These neurons were split into 2 groups, the first ten were active for horizontal bars, and the next 10 for vertical bars.

## 11 Ambiguous visual image 3

One the left you can see the ambiguous image that was shown to the network...

at first the feedback was fully set to "horizontal". Thus, output neuron y2, which is active for the horizontal bar, was very active. 
This task shows how the network can lay its attention on specific parts of the image, due to the feedback it receives.

The intersection point of the 2 graphs is at a firing rate of 55 Hz of the vertical prior neurons. It was expected, that this point would be in the middle at 100 Hz. However, due to the unsupervised way the network learns the ares y2 and y8 do respond to are not of the same size.
y2 responds to an area with a width of 7 pixels, while the response area of y8 was 8 pixels. This means that y8 inherently responds stronger to the same stimulus, thus the intersection point is shifted in its favor. 

## 13 Analysis and simulation of the network 2

On the left you can see the 4 different input images of this experiment. It was defined, that a noiseless input image always has 3 black pixels next to each other. Thus for class 1 the black pixels were at positions 0, 1 and 2. For class 2 they were at postions 2, 3 and 4, and so on. This means that the classes have an overlap of 1 pixel.
The feedback is marked by the red rectangle which is drawn always around the 3 theoretical black pixels of a group.

Under A1 you can see a perfect noiseless input image of class 2 with the feedback also indicating class 2. Under B1 you can see the derived Bayesian posterior in red. In blue you can see the percentage of output spikes that were emitted by each output neuron. As you can see they are the same.
To make it more interesting we also have other noisy input images.
At image 6 you can see the black pixels are positioned between the centers of class 1 and 2. The feedback meanwhile is indication class 4. This results in split probabilities between class 1 and 2, while the biggest probabilities are obtained for class 4. This shows, how the network is choosing a hypothesis for which there is zero visual evidence. Just like in the illusory contour experiment of Lee and Mumford.
At the bottom you can see the obtained Kullback-Leibler divergence for these 6 images. It was 0.01, this value is important as baseline for the next experiments.
