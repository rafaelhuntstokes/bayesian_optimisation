# Scintillation Emission Model Calibration using Bayesian Optimisation

This package uses Bayesian Optimisation to efficiently identify the best set
of parameters to maximise the agreement between data and a Monte-Carlo simulation.

## Introduction

Particle interactions in SNO+ are observed as spatial and temporal distributions 
of photomultiplier tube (PMT) hits. In order to reconstruction the position of particle
interactions in the scintillator, an accurate model of the observed PMT hit time
distribution must be obtained. 

This optimisation package was developed to improve the efficiency of calibrating
the SNO+ scintillation photon emission timing model. Scintillation photons are
generated according to a probabilty distribution, given empirically by:

$$ P(t) = \sum_{i=1}^4 A_i \frac{e^{-t/t_i} - e^{-t/t_r}}{t_i - t_r} $$

The shape of this distribution is governed by the time constants, $t_i$, rise-time, $t_r$, 
and the relative weightings, $A_i$. These parameters must be optimised.

Unfortunately, the observed PMT hit time distibutions are dependent not just on the 
scintillator emission time distribution. Other processes, such as scattering, reflections
and the PMT responses themselves are convolved with the scintillator emission timing.
Thus, it is impossible to create an analytic relationship between the timing parameters
and the data - MC agreement. Moreover, to simulate with a given combination of 
timing parameters and obtain the data - MC agreement is very computationally 
expensive.

Since we have a high-dimensional expensive to evaluate black-box function (the data - MC agreement
as a function of the timing parameters), exhaustive methods, such as gridsearch,
take a prohibitive amount of time to identify good solutions. However, this is the
problem statement Bayesian optimisation is best suited for!

## Bayesian Optimisation with Gaussian Processes
Bayesian optimisation leverages previous samples of the objective function to
select the next best combination of parameters to sample at. A cheap to evaluate
approximation of the objective is used (the 'surrogate') alongside an acquisition
function to select the next parameters.

### The Surrogate: Gaussian Process
The surrogate function is a cheap to evaluate approximation of the objective. In
this case, we use a Gaussian process as the surrogate.

A Gaussian process is a probability distribution over a family of functions, defined
by a mean, $\mu$, and covariance, $\Sigma$, function. The mean function gives the
expected value of the Gaussian process at a given point, $\vec{x}$, and the covariance
function provides an estimate of the uncertainty on the mean. Thus, we model each
point in the domain as a multivariate Gaussian distribution, with dimensionality
equal to the number of features (timing parameters) under consideration.

The covariance function, $\Sigma$, determines the family of functions over which
the Gaussian process is defined. $\Sigma$ defines how function values vary as a function
of distance. For example, the Radial Basis Kernel (RBF) encodes smoothly varying
functions as a function of separation:

$$ \text{RBF}(x, x') = exp\left(-\frac{||x - x'||^2}{2l^2}\right)$$

Given two points, $x_1$ and $x_2$, the covariance function is defined as:

$$ \Sigma = \begin{pmatrix} 
                \text{RBF}(x_1, x_1) & \text{RBF}(x_1, x_2) \\ 
                \text{RBF}(x_2, x_1) & \text{RBF}(x_2, x_2) 
            \end{pmatrix}
$$

