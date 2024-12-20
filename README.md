# Scintillation Emission Model Calibration using Bayesian Optimisation

This package uses Bayesian Optimisation to efficiently identify the best set
of parameters to maximise the agreement between data and a Monte-Carlo simulation.

## Introduction

Particle interactions in SNO+ are observed as spatial and temporal distributions 
of photomultiplier tube (PMT) hits. In order to reconstruction the position of particle
interactions in the scintillator, accurate models of the observed PMT hit time
distributions must be obtained. 

This optimisation package was developed to improve the efficiency of calibrating
the SNO+ scintillation photon emission timing model. Scintillation photons are
generated according to a probabilty distribution, given empirically by:

$$ P(t) = \sum_{i=1}^4 A_i \frac{e^{-t/t_i} - e^{-t/t_r}}{t_i - t_r} $$

The shape of this distribution is governed by the time constants, $t_i$, rise-time, $t_r$, 
and the relative weightings, $A_i$. These parameters must be optimised.

Unfortunately, the observed PMT hit time distibutions are dependent not just on the 
scintillator emission time distribution. Other processes, such as scattering, reflections
and the PMT responses themselves are convolved with the scintillator emission timing.
Thus, it is 