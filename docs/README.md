# Scintillation Emission Model Calibration using Bayesian Optimisation

This package uses Bayesian Optimisation to efficiently identify the best set
of parameters to maximise the agreement between data and a Monte-Carlo simulation.

It was developed to significantly reduce the time necessary to tune these models,
relative to the existing gridsearch method. In addition, it was desirable to create
a fully automated workflow, which would execute with a single button.
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
this case, we use a Gaussian process (GP) as the surrogate.

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

$$ \text{RBF}(x, x') = exp\left(-\frac{\lVert x - x'\rVert ^2}{2l^2}\right)$$

Where $l$ is a tunable length scale parameter. Given two points, $x_1$ and $x_2$, 
the covariance function is defined as:

$$ \hat{K}_{1,2} = \begin{pmatrix} 
\text{RBF}(x_1, x_1) & \text{RBF}(x_1, x_2) \\ 
\text{RBF}(x_2, x_1) & \text{RBF}(x_2, x_2) \\
\end{pmatrix}
+ \delta \mathcal{I}
$$

It is best practice to add a small noise term to the diagonal, $\delta \mathcal{I}$. This is used to
model measurement error and ensure the matrix remains positive-definite, which is
essential for the matrix inversions calculated in the posterior update rules below.

![image](docs/gp_samples.png)

The image above shows the mean function (black line) and associated error (red) band
as a function of $t_1$. The grey functions are examples of samples of individual
functions instantiated from the infinite family of functions defined by the GP. Each
subfigure shows different families of functions, as defined by the RBF (left) and
Matern (right) covariance functions.

#### Prior and Posterior: the Bayesian in Bayesian Optimisation
As more objective function values are sampled, the Bayesian optimiser iteratively
updates its best guess of the form of the objective function. This is done by
calculating the conditional probability distribution in light of the previous
samples. 

This highlights another reason for choosing a Gaussian process as the surrogate:
the conditional probability distribution remains Gaussian, with an updated mean
and covariance function. Thus, to compute the posterior, we only need to update
the mean and covariance functions of the Gaussian process.

For convenience, we start each iteration with a zero mean prior across the domain.
In this case, the update rules are:

$$ \mu'(\vec{x}) = \hat{K}_{m,p}(\hat{K}_{m,m} + \delta \mathcal{I})^{-1}\vec{F}$$

$$\hat{\Sigma}' = \hat{K}_{p,p} - \hat{K}_{p,m}(\hat{K}_{m,m} + \delta \mathcal{I})^{-1}\hat{K}_{m,p} $$

Where $\mu'$ and $\hat{\Sigma}'$ are the mean and covariance function defining the
posterior Gaussian process, $\hat{K}_{x,y}$ are the covariance matrices between
measured and/or predicted points and $\vec{F}$ is a vector of objective function values
at each measured point.

### The Acquisition Function: Lower Confidence Bound
With the posterior in hand, the acquisition function is evaluated to choose the
next objective function sample point. This work uses the Lower Confidence Bound
(LCB) acquisition function due to its simplicity:

$$\text{LCB}(\vec{X}) = \mu(\vec{X}) - \lambda \sigma(\vec{X})$$

Where $\mu(\vec{X})$ is the mean function of the GP surrogate at each point $\vec{X}$, 
$\sigma(\vec{X})$ is the respective uncertainty and $\lambda$ is a tunable parameter
determining the emphasis placed on exploitation or exploration. For small values of 
$\lambda$, exploitation of previously found good solutions is maximised. This
encourages faster convergence, with the risk of becoming stuck in a local minimum.
With larger values of $\lambda$, exploration is prioritised. This leads to slower
convergence, but with greater probability of finding the global best solution.

In any case, the next objective sample point is found by minimising the LCB.

## Quickstart: Running Algorithm

Requirements:
1. python 3.8 or above
2. python libraries: `numpy`, `scipy`, `matplotlib`, `json`, `string`, `copy`, `os`
3. `RAT` MC simulation software
4. HTCondor computing cluster software

Running instructions:
1. Clear previous log files, template macros and output plots: `./clean_files.sh`
2. Reset `opto_log.JSON` by copying `opto_log_CLEAN.JSON` into `opto_log.JSON`
3. Create template DAG files: `python3 run_algo.py`
4. Submit condor DAG to cluster: `condor_dag_submit dag_running/main.dag`

In addition, paths for `RAT` MC software and output files, plots, macros, measured points arrays must
be updated. 

## Algorithm Implementation
In an standard calibration, it is necessary to have four terms in the scintillation
emission time PDF. This leads to a challenging nine-dimensional optimisation, with
parameters:

$$ \vec{X} = \begin{pmatrix} 
                t_1 \\ 
                t_2 \\
                t_3 \\
                t_4 \\
                t_r \\
                A_1 \\
                A_2 \\
                A_3 \\
                A_4
             \end{pmatrix}$$
And the constraint $ \sum_i A_i = 1$.

Given that the surrogate must be defined upon a sufficiently fine mesh of points
to capture local variations in the objective, it is infeasible
to perform the optimisation across all parameters simultaneously. Instead, parameters
are optimised in pairs, whilst holding the others constant.

Since changing any given parameter alters the environment within which other parameters are
tuned, the algorithm repeats each tuning after updates to other parameters, until
convergence is reached.

![image](docs/algo_loops.png)

The diagram shows how pairs of parameters are tuned, before moving to tuning their
respective amplitudes. Once the amplitudes are tuned, the algorithm returns to tuning
the decay constants, which now experience a different environment. This recursive
tuning continues until a stable solution is reached for both the decay constants
and weights.

After the two pairs of parameters are tuned, the rise time optimisation begins.
Since the rise time appears in every term in the emission time PDF, it is tuned
separately. Again, the algorithm returns to the first pair tuning afer the rise
time converges, and this recursion continues until all parameters have converged.

Convergence is defined as a < 5 % change in parameter value relative to the previous
iteration.

### Scripts
There are four main parts to the Bayesian optimiser:
1. Selecting next sample parameters
2. Simulating MC with selected parameters
3. Calculating observed time profiles and evaluating data - MC fit
4. Checking if convergence criteria met

#### Selecting Next Sample

INPUTS : previously measured points (if any) \
OUTPUTS: combination of parameters to sample objective with next and `RAT` MC macros 

`select_parameters,py` is the main script of this stage. If the first iteration,
a random choice of parameter values is chosen. Else, methods from `point_selector.py`
are inherited and used to calculate the posterior and maximum of the LCB acquisition
function. 

Definitions of the feature space for each parameter are defined within `select_parameters`.
`select_parameters.py` loads information on the current iteration from `opto_log.JSON`,
a JSON dictionary structure which tracks the current parameters, global bests, convergence
criteria and iteration number. Based on this information, and any previously measured
objective values stored in the appropriate measured_points `.npy` array, the next
combination of parameters to simulate with are determined. Finally, `select_parameters.py`
creates the respective simulation macro files for use by the `RAT` MC simulation
software.

`point_selector.py` contains the core methods of the Bayesian optimiser, including
functions to compute the posterior mean and covariance function based on previously
measured points. In addition, the LCB acquisition function is implemented here.

![image](docs/algo_output.png)

The image above shows the surrogate (left), uncertainty (middle) and acquisition function
(right) for the $(t_1, t_2)$ optimisation loop.

#### Simulating MC with Selected Parameter Values

INPUT: macro file \
OUTPUT: RATDS MC simulation files

Given a combination of scintillation emission time parameters, particle interactions
are simulated. The simulation uses the SNO+ Reactor Analysis Toolkit (RAT) MC software,
which simulates the initial particle interactions, detector geometry, hardware, scintillator model, propagation of scintillation light and its subsequent detection.

By default, the MC simulates <sup>214</sup>Bi $\beta$/$\gamma$ radioactive decays
within the scintillator.

The output RATDS files contain the reconstructed position and time of each simulated
interaction.

#### Calculating the Observed Scintillator Time Profiles

INPUT: RATDS MC simulation files \
OUTPUT: observed scintillator time profile, $\chi^2$ between data and MC

`time_residuals.py` computes the observed scintillator time profiles ('time residuals')
using the following:

$$ t_{res} = t_{hit} - t_{tof} - t_{ev}$$

Where $t_{hit}$ is the PMT hit time, $t_{tof}$ is the calculated time of flight between
the hit PMT and reconstructed interaction position, and $t_{ev}$ is the reconstructed
interaction time. By formulating the residual time, $t_{res}$, as above, first order
position dependence is removed from the observed time profiles, allowing a fair
comparison between events throughout the scintillator volume. As stated above,
the observed time residual distributions are functions of the scintillator emission
timing, PMT responses and other convolved optical properties. Assuming these other
processes are well modelled, the scintillator emission time model is optimised to
maximise the agreement between the data and simulated time residual distributions.

The agreement between data and MC is quantified by calculating the $\chi ^2$, with the MC
distribution first normalised to the counts in the data.

$$ \chi ^2 = \sum_{bins} \frac{(O_i - M_i)^2}{M_i}$$

Where $O_i$ and $M_i$ represent the observed and measured frequencies, respectively in bin $i$.

The $\chi ^2$ agreement is saved in an updated `measured_points.npy` array for subsequent
use in `select_parameters.py`.

![image](docs/time_residuals.png)

The image above shows the observed time residual distributions for the data (black) and
MC simulations using the Bayesian optimisation result (green), grid search (red) and
a timing model measured on the benchtop (blue).
#### Termination Criteria

There are three termination scripts:

1. `terminate_opto.py` - determine if convergence or max iters reached for given parameter
optimisation block
2. `terminate_block.py` - determine if convergence or max iters reached for time constant - amplitude tuning block 
3. `terminate_algo.py` - determine if convergence or max iters reached for entire algorithm

As stated above, convergence is defined as whether the parameter values change by < 5 % between
iterations. The max iters are specified in the `opto_log.JSON` file.

Each of these terminate scripts are run as post scripts in the condor DAG structure explained below.

## Automatic Relevance Determination
As shown in the RBF kernel equation, there are hyperparameters within the covariance
function itself. In this work, the hyperparameters include the respective length
scales in the RBF kernel, $l$, and the exploration coefficient, $\lambda$, used in
the LCB acquisition function. The exploration coefficient is set to $\lambda = 5$ by default.

Automatic Relevance Determination (ARD) is the method used to automate the optimisation
of the length scale parameters in the RBF kernel. A length scale is defined for each
feature, parametising the sensitivity of the objective to each feature. For example,
points with similar $t_1$ values, but vastly different $t_4$, will still yield similar
objective function samples. Thus, these points should yield large RBF covariance values,
despite the naive $\vec{l} = 1$ kernel suggesting they are very distant points.

ARD is a maximum-likelihood method, where the likelihood
of the surrogate given the sampled data is maximised, as a function of the kernel
length scale parameters. The log-likelihood is given by:

$$ log(\mathcal{L}) = - \frac{1}{2}\vec{F}^T\hat{K}^{-1}\vec{F} - \frac{1}{2}log(det(\hat{K})) - \frac{n}{2}log(2\pi)$$

Where $\vec{F}$ is a vector of measured objective values, $\hat{K}$ is the matrix of
RBF kernel values for the measured points, and $n$ is the number of measured points. 

The first term is the *model-fit term*, which determines whether the covariance structure
predicted by the kernel is found in the measured points. Where the expected covariance
structure is not found in the measured points, this term is maximised.

The second term is the *complexity penalty term*. The determinant of the covariance
matrix is larger for more complex, flexible models, and smaller for more rigid,
less complex models. 

For each iteration, the $log(\mathcal{L})$ is maximised within the `update_surrogate()`
function.

![image](docs/ARD.png)

The image above shows the $log(\mathcal{L})$ distribution obtained within an iteration
of the $(t_1, t_2)$ optimisation loop, as a function of length scale.
## Condor DAG Structure
As noted in the introduction, this package was developed to be fully automated. To this
end, condor's Direct Acyclic Graph (DAG) structures were utilised. DAGs are executed as
a series of sequential nodes, with subsequent nodes only running once the previous
are complete.

![image](docs/dag_structure.png)

As the image shows, there are a number of nested dags. This structure was necessary to
get around the 'acyclic' part of the direct acyclic graph structure: i.e., acyclic
means without loops. However, loops are necessary in order to recursively tune the
parameters. In order to handle loop structures in the algorithm, post-scripts, which
run at the end of each DAG file, were added. These post scripts are shown in 
gray boxes in the diagram.

Given the completion of a DAG, the post script runs and returns a boolean pass or
fail. Upon a fail, the parent DAG is resubmitted. By using post-scripts to evaluate
convergence or max iters reached after each DAG, the algorithm may simulate loop
behaviour.

Note: the structure of `first_pair.dag`, `second_pair.dag` and `rise_time.dag` are
equivalent, containing `opto.dag` in each case. All that changes between these blocks
are the parameters to be optimised, which is tracked in the `opto_log.JSON` file and
loaded in `select_parameters.py` each iteration.

Finally, it is worth noting the MC step is carried out as a series of
parallel simulation jobs. `select_parameters.py` creates a single simulation macro,
which is executed as a number of identical jobs on the cluster. This makes full use
of the available parallel compute infrastructure available and speeds the MC bottleneck.

## Future Work and Customisation
Future calibrations of the SNO+ scintillation emission model are inevitable. Whenever
the composition of the scintillator is changed, the scintillation model requires
re-calibrating. This section briefly outlines the steps necessary to update the
optimiser to handle new scintillator compositions, e.g. following tellurium double-beta
decay isotope loading.

### Updates for Te Loading
The main changes necessary are:
1. Update dataset array used in `time_residuals.py` for $\chi^2$ calculations. See my thesis
for details on collecting a new dataset (Chapter 3)
2. Change liquid scintillator material / optics table specificed in `RAT` MC macro file.
This is simply achieved by altering the string in `select_parameters.py` 

### Customisation and Algorithm Improvements
This work uses the RBF kernel for the covariance function evaluation, and the LCB
acquisition function. These were chosen for their simplicity, but are not necessarily
the best choices for our dataset. In the future, it may be desirable to compare
performance using other kernels, such as the Matern, and/or other acquisition functions.

For example, the Matern kernel encodes a less smoothly varying family of functions than
the RBF, which may well be more approprate for these data. In addition, more sophisticated
acquisition functions, such as expected improvement or probabaility of improvement,
may converge faster.

To add these functions to the Bayesian optimiser package, one simply needs to add
them as functions in the `PointSelector` class, within `point_selector.py`. Once added,
the new method call must be updated within the `PointSelector` class' `update_surrogate()`
method. For the acquisition function, the appropriate lines in `select_parameters.py` should
be changed to call the new acquisition function.