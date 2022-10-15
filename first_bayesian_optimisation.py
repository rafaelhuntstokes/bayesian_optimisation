import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from MONTE_CARLO_SAMPLER import GenerateDataset

def acquisition_PROBABILITY_IMPROVEMENT(means, sigmas, global_best, epsilon):
    return 1- norm.cdf(-(global_best + epsilon - means) / sigmas)

def acquistion_EXPECTED_IMPROVEMENT(means, sigmas, global_best, epsilon):
    return np.where(sigmas > 0, (epsilon + global_best - means)* norm.cdf((global_best+epsilon-means)/sigmas) + sigmas * norm.pdf((global_best+epsilon-means)/sigmas), np.zeros(means.shape))

def kernel_4d(measurements, predictions, scale):
    # reshape the input values to the correct matrix sizes
    measurements = measurements[:, None, :]
    predictions = predictions[None, :, :]    
    exponent = np.linalg.norm(measurements - predictions, axis = 2)
    rbf = (scale[0]**2) * np.exp(-exponent**2 / (2*scale[1]**2))
    # add some noise (accounts for the initial prior with zero measurements) and also maybe add some measurement noise?
    if rbf.shape[0] == rbf.shape[1]:
        diagonal= np.eye(np.shape(rbf)[0], np.shape(rbf)[1])
        noise = 0.02**2 * diagonal
        return rbf + noise
    else:
        return rbf

def gaussian_process_4d(measured_pts, cost, predicted_pts, kernel_params):
    # use the kernel to find the covariance matrix elements
    # first, the covariance of the observations - N x N square matrix , N = number of observations
    cov_11 = kernel_4d(measured_pts, measured_pts, kernel_params)

    # covariance between measured and predicted points - N x M  matrix,  M = number of predicted points
    cov_12 = kernel_4d(measured_pts, predicted_pts, kernel_params)
    
    # covariance of the predicted points (this is a square matrix) - M x M square matrix, M = number of predicted points
    cov_22 = kernel_4d(predicted_pts, predicted_pts, kernel_params)
    
    # only invert and multiply sig_21 x sig_22 ^-1 once ...
    chunk_solved = (np.linalg.inv(cov_11) @ cov_12).T
    mu_predicted = chunk_solved @ cost
    cov_predicted = cov_22 - (chunk_solved @ cov_12)

    return mu_predicted, cov_predicted



ITERS = 100
# amplitude of entire kernel, then std^2 of the input features: {t1, t2, A1, A2}
SCALE = [1, 0.5]


"""
HERE WE DEFINE THE MOCK TIME RESIDUAL DATA ARRAYS ETC.
"""
# generate fake measured data from detector with "true" parameters t1 = 5.0, t2 = 15.0, A1 = 0.8, A2 = 0.2
# and N sampled data points
N = 20000
# detector_data = generate_dataset(5.0, 15.0, N)
t1_true = 5
t2_true = 15
A1_true = 0.8
A2_true = 0.2
detector_data = GenerateDataset(t1_true, t2_true, A1_true, A2_true, 0, 150, N).residuals
# create a histogram of this dataset. The bin contents will be used to evaluate the MSE of each proposed
# solution from Bayesian optimizer
binning = np.arange(0, 151, 1)
data_counts, _ = np.histogram(detector_data, bins = binning, density = True)

t1_res = 1
t2_res = 1
A1_res = 0.1
A2_res = 0.2
t1_axis = np.arange(0.1, 5.0, t1_res)
t2_axis = np.arange(10.0, 30.0, t2_res)
A1_axis = np.arange(0.01, 1.0, A1_res)
A2_axis = np.arange(0.01, 1.0, A2_res)

NUM_FEATURES = 4 # t1, t2, A1, A2
measured_pts = np.zeros((ITERS+1, NUM_FEATURES), dtype = np.float16) # square array, each row is a "measurement", and each column is the value of the features of measurement
predicted_pts = np.zeros((len(t1_axis), len(t2_axis), len(A1_axis), len(A2_axis), 4), dtype =np.float16)
for (i, t1_val) in enumerate(t1_axis):
    for (j, t2_val) in enumerate(t2_axis):
        for (k, A1_val) in enumerate(A1_axis):
            for (l, A2_val) in enumerate(A2_axis):
                predicted_pts[i,j,k,l,0] = t1_val
                predicted_pts[i,j,k,l,1] = t2_val
                predicted_pts[i,j,k,l,2] = A1_val
                predicted_pts[i,j,k,l,3] = A2_val
predicted_pts = predicted_pts.reshape(len(t1_axis) * len(t2_axis) * len(A1_axis) * len(A2_axis), 4)
print(predicted_pts.shape)
cost = np.zeros(ITERS+1) #track the cost function for each iteration

"""
TIME RESIDUAL FAKE DATASETS DEFINED: READY TO PASS TO OPTIMISER!
"""

convergence_count = 0
last = None

bests = []
t1_best = []
t2_best = []
A1_best = []
A2_best = []
global_best = 1e6
for iter in range(ITERS):
    print(f"########### iter {iter} ###########")
    mu_predicted, cov_predicted = gaussian_process_4d(measured_pts, cost, predicted_pts, SCALE)
    std_predicted = np.sqrt(abs(np.diag(cov_predicted)))

    epsilon = 0.5
    expectedImprovement = acquistion_EXPECTED_IMPROVEMENT(mu_predicted, std_predicted, global_best, epsilon)
    max_improvement_idx = np.argmax(expectedImprovement)
    
    # maximimise the expected improvement and select it as the next iterations sampling point
    max_idx = predicted_pts[max_improvement_idx, :]
    print(f"Best improvement at pred point: {max_idx}")

    # find what position this goes to
    sample_position = [max_idx[0], max_idx[1], max_idx[2], max_idx[3]]
    if sample_position == last:
        convergence_count +=1
    else:
        convergence_count = 0
    last = sample_position
    if convergence_count == 5:
        print(f"Converged in {iter} iterations.\nGlobal Best Position: ({sample_position[0]},{sample_position[1], sample_position[2], sample_position[3]})")#\nError of minimum: {np.sqrt(sample_position[0]**2 + sample_position[1]**2)}")
        break

    measured_pts[iter, 0] = sample_position[0]
    measured_pts[iter, 1] = sample_position[1]
    measured_pts[iter, 2] = sample_position[2]
    measured_pts[iter, 3] = sample_position[3]

    magnitude = sample_position[2] + sample_position[3]
    sample_position[2] /= magnitude
    sample_position[3] /= magnitude
    
    model = GenerateDataset(sample_position[0], sample_position[1], sample_position[2], sample_position[3], 0, 150, N).residuals
    # evaluate cost function by histogramming model and finding sum of squared diff between bins
    model_counts, _ = np.histogram(model, bins = binning, density = True)
    goodness_of_fit = np.sum((model_counts - data_counts)**2)

    iter_best = goodness_of_fit
    if iter_best < global_best:
        global_best = iter_best
        t1_best.append(sample_position[0])
        t2_best.append(sample_position[1])
        A1_best.append(sample_position[2])
        A2_best.append(sample_position[3])
        plt.figure()
        plt.hist(model, bins = binning, density = True, label = f"Iter: {iter} | t1: {sample_position[0]} | t2: {sample_position[1]} | A1: {sample_position[2]} | A2: {sample_position[3]}", histtype = "step")
        plt.hist(detector_data, bins = binning, density = True, label = f"Detector Data | t1: {t1_true} | t2: {t2_true} | A1: {A1_true} | A2: {A2_true}", histtype = "step")
        plt.legend()
        plt.ylim((0,0.2))
        plt.savefig(f"frames/4d_time_residuals/{iter}.png")
        plt.close()
    bests.append(global_best)
    

    print(f"Global Best Solution: ({t1_best[-1]} {t2_best[-1]} {A1_best[-1]} {A2_best[-1]})")

    cost[iter] = goodness_of_fit

plt.plot(bests)
plt.yscale("log")
plt.show()