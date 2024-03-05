# Normalization condition

nessler EQ 10:
summe von e hoch w = 1

fire_prob = e hoch u 
u = w * y

... haaa?

I glaub i hab verstanden wo die Normalization fehlt!...
bei den output neuronen verwenden wir soft-WTA (adaptive inhibition)... und das normalisiert alles...
ABER bei den Input und prior neuronen gibts keine Normalization und das erzeugt mein Problem... wenn ich 10 input neuronen auf einen output habe sollte es gleich viele spikes empfangen, wie wenn ich 100 input neuronen habe??? .... RESEARCH for DISCUSSION

# check this
This result is then transformed into an instantaneous
firing rate, assuming an exponential relationship between rate and
the membrane potential [38]

# Einfach interessant (https://link.springer.com/chapter/10.1007/978-3-319-91908-9_11):

## visual cortex:

The foundational understanding of the apparent power of deep learning is an important current challenge for Theoretical Computer Science. How does this quest relate to the brain? We refer to [73] for a discussion of related literature. Deep learning of some sort does happen in the brain (consider the visual cortex and the hierarchical processing through its areas, from V1 to V2 and V4 all the way to MT and beyond). But there are differences, and perhaps the most fundamental among them is the existence of lateral and backward connections between brain areas. What is their function, and how do they enhance learning?

## Neuronal assemblies

anscheinend neue erkenntnis:
Neuronen bilden assemblies (gruppen) die meistens zusammen feuern, und die daher wahrscheinlich (not proven) stark interconnected sind (horizontally).
Vielleicht performed das gehrin Flächen/Volumen Operationen... AOE Damage (oder auch nicht AOE, sondern verteilter über das areal :/)... neural computation model dafür fehlt!
Eigentlich machts eh sinn, bei gewissem input lernen halt manche neuronen das task... und daher feuern die immer wenn der input daherkommt. Und beim überlappen feuern halt beide neuronengruppen... Und da das netzwerk net weiß dass das grad "2" inputs sind lernt es so als obs eins wär, und daher korrelieren die beiden Gruppen nun.

## memory

zeigt probanden bilder... zb bei eifelturm spikte immer das gleiche neuron, und bei anderen bildern nie. als man eifelturm mit anderem bild kombinierte spikte es danach auch wenn man das andere bild herzeigte... Verbindung hat sich gebildet => neuronal assemblies "bleed" into each other. (they overlap, dont form a new vertical area)
Sie haben da a paar 100 random neuronen vom medial temporal lobe gemessen mit random bildern => 10000e neuronen responden konsistent auf einen Stimulus.

## we assume that the population of excitatory neurons is randomly and sparsely connected, a reasonable model in view of experimental data 

## brain is active, independent of input

A further surprising feature of brain activity is that it is not input driven: the brain is almost as active when there is (seemingly) nothing to compute. For example, the neurons in the primary visual cortex (area V1) are almost as active as during visual processing as they are in complete darkness [64]. Since brain activity consumes a fair portion of the energy budget of an organism, it is unlikely that this spontaneously ongoing brain activity is just an accident, and highlights a clear organizational difference between computers and brains. A challenge for theoretical work is to understand the role of spontaneous activity in brain computation and learning.

# new papers

The neural dynamics of hierarchical Bayesian causal inference in multisensory perception
https://www.nature.com/articles/s41467-019-09664-2   

The Anatomy of Inference: Generative Models and Brain Structure
https://www.frontiersin.org/articles/10.3389/fncom.2018.00090/full#:~:text=To%20infer%20the%20causes%20of,about%20how%20we%20will%20act.

Hierarchical Bayesian Inference and Learning
in Spiking Neural Networks (das basiert auf Nessler!)
https://sci-hub.st/10.1109/TCYB.2017.2768554

Neural substrate of dynamic Bayesian inference in the cerebral cortex
https://www.nature.com/articles/nn.4390