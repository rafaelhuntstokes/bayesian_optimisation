import chunk
from random import sample
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, rv_continuous
import more_itertools
from matplotlib import cm
from MONTE_CARLO_SAMPLER import GenerateDataset

def generate_dataset(t1, t2, N):
    """
    Function creates a mock scintillator emission time PDF and returns sampled values.
    
    INPUTS: t1, t2 - floats, the decay time constants of the preposed model
            N      - int, the number of measurements to sample from generated PDF
    
    RETURN: time_residuals - array of floats, representing "measured" scintillator time residuals.
    """

    class model(rv_continuous):
        def _pdf(self, t, t1, t2):
            A1 = 0.8 # amplitude 1 
            A2 = 0.2 # amplitude 2
            tr = 0.8 # rise time in ns 
            return (A1 * (np.exp( -( t/t1 )) - np.exp(-( t/tr ) ) )  / ( t1 - tr ) ) +  (A2 * (np.exp( -( t/t2 ) ) - np.exp( -( t/tr ) )) / ( t2 - tr ) )
    def cdf():
        pass
    # create the distribution object and sample N times
    # the PDF is valid for the domain 0 --> 50 ns. It is undefined and shit goes wrong for t < 0. 
    dist = model(a = 0, b = 50)
    sample = dist.rvs(t1, t2, size = N)

    return sample

def objective(X):
    return 3*np.sin(X) + 4*np.cos(X)

def dad(X):
    return X**3 - 11

def ackley_1d(x, y=0):
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) 
           - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
           + np.e + 20)

def ackley_2d(X, Y, a, b, c):
    return -a * np.exp(-b * np.sqrt((1/2) * (X**2 + Y**2))) - np.exp((1/2) * (np.cos(c*X)+ np.cos(c*Y))) + a + np.e

def acquisition_PROBABILITY_IMPROVEMENT(means, sigmas, global_best, epsilon):
    inputs = (global_best - means) / sigmas
    print("input to cdf: ", inputs)
    return 1- norm.cdf(-(global_best + epsilon - means) / sigmas)

def acquistion_EXPECTED_IMPROVEMENT(means, sigmas, global_best, epsilon):
    # print(f"Shape Means: {means.shape}\nShape Sigmas: {sigmas.shape}\nShape Global Best: {global_best.shape}")
    norm_cdf = norm.cdf((global_best+epsilon-means)/sigmas)
    norm_pdf = norm.pdf((global_best+epsilon-means)/sigmas)
    # print(f"PDF Shape: {norm_pdf.shape}\n CDF Shape: {norm_cdf.shape}")
    checker = (epsilon + global_best - means)* norm.cdf((global_best+epsilon-means)/sigmas) + sigmas * norm.pdf((global_best+epsilon-means)/sigmas)
    return np.where(sigmas > 0, (epsilon + global_best - means)* norm.cdf((global_best+epsilon-means)/sigmas) + sigmas * norm.pdf((global_best+epsilon-means)/sigmas), np.zeros(means.shape))

def kernel(x_measured, x_predicted, scale):

    # reshape the input values to the correct matrix sizes
    x_measured = x_measured[:, None]
    x_predicted = x_predicted[None, :]
    # print(f"Exponent in the kernel: {(-1/(2*scale[1]**2) * (x_measured - x_predicted)**2)}\nAmplitude of kernel: {scale[0]**2}")

    rbf = (scale[0]**2) * np.exp((-1/(2*scale[1]**2) * (x_measured - x_predicted)**2))

    # add some noise (accounts for the initial prior with zero measurements) and also maybe add some measurement noise?
    if rbf.shape[0] == rbf.shape[1]:
        diagonal= np.eye(np.shape(rbf)[0], np.shape(rbf)[1])
        noise = 0.02**2 * diagonal
        # print(f"ADDED NOISE!\n{noise}\n")
        return rbf + noise
    else:
        # print("NO NOISE!")
        return rbf

def kernel_2d(x1_measured, x1_predicted, x2_measured, x2_predicted, scale):

    # reshape the input values to the correct matrix sizes
    x1_measured = x1_measured[:, None]
    x1_predicted = x1_predicted[None, :]
    x2_measured = x2_measured[:, None]
    x2_predicted = x2_predicted[None, :]
    # print(f"Exponent in the kernel: {(-1/(2*scale[1]**2) * (x_measured - x_predicted)**2)}\nAmplitude of kernel: {scale[0]**2}")

    rbf = (scale[0]**2) * np.exp((-1/(2*scale[1]**2) * (x1_measured - x1_predicted)**2) - (1/(2*scale[2]**2 ) * (x2_measured - x2_predicted)**2))

    # add some noise (accounts for the initial prior with zero measurements) and also maybe add some measurement noise?
    if rbf.shape[0] == rbf.shape[1]:
        diagonal= np.eye(np.shape(rbf)[0], np.shape(rbf)[1])
        noise = 0.02**2 * diagonal
        # print(f"ADDED NOISE!\n{noise}\n")
        return rbf + noise
    else:
        # print("NO NOISE!")
        return rbf

def gaussian_process(x_measured, y_measured, x_predicted, kernel_params):
    """
    Will return the predicted mean and std for every point in the objective domain sampled.
    """
    # use the kernel to find the covariance matrix elements
    # first, the covariance of the observations - N x N square matrix , N = number of observations
    cov_11 = kernel(x_measured, x_measured, kernel_params)
    # print(f"Shape of COV_11: \n{cov_11.shape}\n")
    # print(f"COV_11: \n{cov_11}\n")
    # covariance between measured and predicted points - N x M  matrix,  M = number of predicted points
    cov_12 = kernel(x_measured, x_predicted, kernel_params)
    # print(f"Shape of COV_12: \n{cov_12.shape}\n")
    # print(f"COV_12: \n{cov_12}\n")
    # covariance of the predicted points (this is a square matrix) - M x M square matrix, M = number of predicted points
    cov_22 = kernel(x_predicted, x_predicted, kernel_params)

    # only invert and multiply sig_21 x sig_22 ^-1 once ...
    chunk_solved = (np.linalg.inv(cov_11) @ cov_12).T

    mu_predicted = chunk_solved @ y_measured
    # print(f"Y MEASURED: {y_measured}")
    # print(f"Shape of mu_predicted: \n{mu_predicted.shape}\n")
    # print(f"mu_predicted: \n{mu_predicted}\n")
    cov_predicted = cov_22 - (chunk_solved @ cov_12)

    return mu_predicted, cov_predicted

def gaussian_process_2d(y_measured, x1_measured, x1_predicted, x2_measured, x2_predicted, kernel_params):
    """
    Will return the predicted mean and std for every point in the objective domain sampled.
    """
    # use the kernel to find the covariance matrix elements
    # first, the covariance of the observations - N x N square matrix , N = number of observations
    cov_11 = kernel_2d(x1_measured, x1_measured, x2_measured, x2_measured, kernel_params)
                       
    # covariance between measured and predicted points - N x M  matrix,  M = number of predicted points
    cov_12 = kernel_2d(x1_measured, x1_predicted, x2_measured, x2_predicted, kernel_params)
    # print(f"COV_12: \n{cov_12}\n")
    # covariance of the predicted points (this is a square matrix) - M x M square matrix, M = number of predicted points
    cov_22 = kernel_2d(x1_predicted, x1_predicted, x2_predicted, x2_predicted, kernel_params)

    # only invert and multiply sig_21 x sig_22 ^-1 once ...
    chunk_solved = (np.linalg.inv(cov_11) @ cov_12).T
    mu_predicted = chunk_solved @ y_measured

    cov_predicted = cov_22 - (chunk_solved @ cov_12)

    return mu_predicted, cov_predicted

# initialise a flat Gaussian prior on this domain
ITERS = 100
SCALE = [1, 1, 5]

X1_RES = 50
X2_RES = 50
#  arrays to hold 2D domain and measured position at each iteration
x1_predicted_axis = np.linspace(-4, 4, X1_RES)
x1_measured = np.zeros(ITERS+1)
x2_predicted_axis = np.linspace(-4, 4, X2_RES)
x2_measured = np.zeros(ITERS+1)
print(f"Num grid points is: {len(x1_predicted_axis)} x {len(x2_predicted_axis)} = {len(x1_predicted_axis) * len(x2_predicted_axis)}")

# create a grid in each of the predicted axis
x1_predicted, x2_predicted = np.meshgrid(x1_predicted_axis, x2_predicted_axis)
x1_predicted = x1_predicted.flatten()
x2_predicted = x2_predicted.flatten()

# define true function values to sample from over the domain
ackley_results = ackley_2d(x1_predicted, x2_predicted, 20, 0.2, 2*np.pi).flatten()

# array is 1D and contains the sampled ackley 2D values for each iteration: input (x1, x2) --> y
y_measured = np.zeros(ITERS+1)


"""
HERE WE DEFINE THE MOCK TIME RESIDUAL DATA ARRAYS ETC.
"""
# generate fake measured data from detector with "true" parameters t1 = 5.0, t2 = 15.0 
# and N sampled data points
N = 15000
# detector_data = generate_dataset(5.0, 15.0, N)
detector_data = GenerateDataset(5, 15, 0, 150, N).residuals
# create a histogram of this dataset. The bin contents will be used to evaluate the MSE of each proposed
# solution from Bayesian optimizer
binning = np.arange(0, 151, 1)
print(binning)
data_counts, _ = np.histogram(detector_data, bins = binning, density = True)

# create the axes for t1 and t2 parameters to be tuned over
t1_res = 0.5
t1_axis = np.arange(0.1, 10, t1_res)

t2_res = 1.0
t2_axis = np.arange(0.1, 30, t2_res)

# meshgrid the axes to create the flat predicted value arrays for use in Bayesian Optimiser
t1_predicted, t2_predicted = np.meshgrid(t1_axis, t2_axis)
t1_predicted = t1_predicted.flatten()
t2_predicted = t2_predicted.flatten()
print(t1_predicted.shape)
# create arrays to store the measured sum SE for a given sample, and the values of t1, t2 used
t1_measured = np.zeros(ITERS+1)
t2_measured = np.zeros(ITERS+1)
cost = np.zeros(ITERS+1)

"""
TIME RESIDUAL FAKE DATASETS DEFINED: READY TO PASS TO OPTIMISER!
"""


convergence_count = 0
last = None

bests = []
t1_best = []
t2_best = []
global_best = 1e6
for iter in range(ITERS):
    print(f"########### iter {iter} ###########")
    # mu_predicted, cov_predicted = gaussian_process_2d(y_measured[:iter], x1_measured[:iter], x1_predicted.flatten(), x2_measured[:iter], x2_predicted.flatten(), SCALE)
    mu_predicted, cov_predicted = gaussian_process_2d(cost[:iter], t1_measured[:iter], t1_predicted, t2_measured[:iter], t2_predicted, SCALE)
    std_predicted = np.sqrt(abs(np.diag(cov_predicted)))
    
    # evaluate the acquisition function over the domain
    # min_mu_idx = np.argmin(mu_predicted)

    # obtain the value of t1, t2 at this minimum
    # t1_min = t1_predicted[min_mu_idx]
    # t2_min = t2_predicted[min_mu_idx]

    # print(t1_min, t2_min)
    # pretend to generate Monte Carlo with these parameters (sample PDF)
    # model = generate_dataset(t1_min, t2_min, N)
    # evaluate cost function by histogramming model and finding sum of squared diff between bins
    # model_counts, _ = np.histogram(model, bins = binning, density = True)
    # goodness_of_fit = np.sum((model_counts - data_counts)**2)

    # this is the FIRST sampling that has happened in this iteration. Update arrays.
    # t1_measured[iter] = t1_min
    # t2_measured[iter] = t2_min
    # cost[iter]= goodness_of_fit

    # iter_best = ackley_results[min_mu_idx]
    # iter_best = goodness_of_fit
    # if iter_best < global_best:
    #     global_best = iter_best
    # bests.append(global_best)
    # print(f"global best position: ({x1_predicted[min_mu_idx], x2_predicted[min_mu_idx]})\nValue: {global_best}")

    epsilon = 1.0
    expectedImprovement = acquistion_EXPECTED_IMPROVEMENT(mu_predicted, std_predicted, global_best, epsilon)
    max_improvement_idx = np.argmax(expectedImprovement)

    # maximimise the expected improvement and select it as the next iterations sampling point
    # sample_position = [x1_predicted[max_improvement_idx], x2_predicted[max_improvement_idx]]
    sample_position = [t1_predicted[max_improvement_idx], t2_predicted[max_improvement_idx]]
    if sample_position == last:
        convergence_count +=1
    else:
        convergence_count = 0
    last = sample_position
    if convergence_count == 5:
        print(f"Converged in {iter} iterations.\nGlobal Best Position: ({sample_position[0]},{sample_position[1]})")#\nError of minimum: {np.sqrt(sample_position[0]**2 + sample_position[1]**2)}")
        break

    # update measurements for each iteration
    # x1_measured[iter] = sample_position[0]
    # x2_measured[iter] = sample_position[1]
    # y_measured[iter] = ackley_results[max_improvement_idx]

    # SECOND sampling in this iteration ... 
    t1_measured[iter] = sample_position[0]
    t2_measured[iter] = sample_position[1]

    # fake the MC again 
    # model = generate_dataset(sample_position[0], sample_position[1], N)
    model = GenerateDataset(sample_position[0], sample_position[1], 0, 150, N).residuals
    # evaluate cost function by histogramming model and finding sum of squared diff between bins
    model_counts, _ = np.histogram(model, bins = binning, density = True)
    goodness_of_fit = np.sum((model_counts - data_counts)**2)

    iter_best = goodness_of_fit
    if iter_best < global_best:
        global_best = iter_best
        t1_best.append(sample_position[0])
        t2_best.append(sample_position[1])
        plt.figure()
        plt.hist(model, bins = binning, density = True, label = f"Iter: {iter} | t1: {sample_position[0]} | t2: {sample_position[1]}", histtype = "step")
        plt.hist(detector_data, bins = binning, density = True, label = f"Detector Data | t1: {5} | t2: {15}", histtype = "step")
        plt.legend()
        plt.ylim((0,0.2))
        plt.savefig(f"frames/2d_time_residuals/{iter}.png")
        plt.close()
    bests.append(global_best)
    

    print(f"Global Best Solution: ({t1_best[-1]} {t2_best[-1]})")

    cost[iter] = goodness_of_fit
    """
    Plotting scripts for each iteration!
    """

    # plt.imshow(ackley_results.reshape((X1_RES, X2_RES)))
    # plt.plot(np.where(x1_predicted_axis == sample_position[0]), np.where(x2_predicted_axis == sample_position[1]), color = "red", label = "sampled position")
    # plt.savefig(f"./frames/2d/{iter}.png")
    # plt.close()

    idx1 = np.digitize(t1_measured[:iter+1], bins = t1_axis) - 1
    idx2 = np.digitize(t2_measured[:iter+1], bins = t2_axis) - 1
    fig, axes = plt.subplots(1,3, tight_layout = True)
    # box = axes[1].get_position()
    # box.x0 = box.x0 + 1
    # box.x1 = box.x1 + 1
    # axes[1].set_position(box)
    # pos1 = axes[0].imshow(ackley_results.reshape((X1_RES, X2_RES)), alpha = 0.8)
    if iter == 1:
        axes[0].scatter(idx1, idx2 ,color = "red")
    axes[0].plot(idx1, idx2, color = "red", linestyle = "--", marker = "o", label = "sampled position")
    axes[0].set_title(f"t1, t2")
    axes[0].legend()

    pos2 = axes[1].imshow(expectedImprovement.reshape((len(t1_axis), len(t2_axis))))
    # plt.colorbar(pos2, fraction=0.046, pad=0.04)
    axes[1].set_title("EI Acquisiton Function")

    axes[2].imshow(mu_predicted.reshape((len(t1_axis), len(t2_axis))))
    axes[2].set_title("Surrogate")
    plt.suptitle(f"Iteration {iter}")
    plt.savefig(f"./frames/fake_data/{iter}.png")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(t1_predicted.reshape((len(t1_axis), len(t2_axis))), t2_predicted.reshape((len(t1_axis), len(t2_axis))), mu_predicted.reshape((len(t1_axis), len(t2_axis))), cmap = cm.coolwarm)
    plt.savefig(f"./frames/fake_data_surrogate/{iter}.png")
    plt.xlabel("t1")
    plt.ylabel("t2")
    plt.close()
plt.plot(bests)
plt.yscale("log")
plt.show()