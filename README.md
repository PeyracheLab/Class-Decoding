# Class-Decoding
Supervised and unsupervised techniques to decode signals from a population of neurons: the case of head-direction cells.

This tutorial will guide through the analysis of HD cell population data and how to extract a head-direction and simple topological features from the population activity. It starts with PCA and then shows how to use IsoMap to extract the topology without the inherent constraints of PCA.

Data are stored in dataHD.mat
what it contains:
 - Q{run,sws and rem} are three binned spike trains from 19 simultaneously
   recorded head-drection (HD) neurons in the anterodorsal nucleus of a
   freely moving mouse. Bin size: 10ms
 - angRun: animal's HD (same time bins as Qrun). Two column matrix (times,
   data)
 - prefHD: preferred HD of each neuron.
 - hdTuning: tuning curves. First column are the angular bin values

For more details regarding the recording:
Peyrache et al. (2015) Internally organized mechanisms of the head direction sense. Nature Neuroscience 18, 569–575.
