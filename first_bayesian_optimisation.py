from random import sample
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import time
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
    """
    Given feature vectors of measured points, sampled cost function at measured points, predicted points, 
    find the mean prior for every predicted point.
    """
    # first, the covariance of the observations, shape (num_measurements, num_measurements)
    cov_11 = kernel_4d(measured_pts, measured_pts, kernel_params)

    # covariance between measured and predicted points, shape (num_measurements, num_predictions)
    cov_12 = kernel_4d(measured_pts, predicted_pts, kernel_params)
    
    # covariance of the predicted points, shape (num_predictions, num_predictions)
    cov_22 = kernel_4d(predicted_pts, predicted_pts, kernel_params)
    
    # solve for posterior mean and covariance for every predicted point
    chunk_solved = (np.linalg.inv(cov_11) @ cov_12).T
    mu_predicted = chunk_solved @ cost
    cov_predicted = cov_22 - (chunk_solved @ cov_12)

    return mu_predicted, cov_predicted

def create_dataset(num_samples, t1, t2, t3, t4, A1, A2, A3, A4, pdf_low=0, pdf_high=150, bin_width=1):
    detector_data = GenerateDataset(t1, t2, t3, t4, A1, A2, A3, A4, pdf_low, pdf_high, num_samples).residuals
    # binning = np.arange(pdf_low, pdf_high+bin_width, bin_width)
    # data_counts, _ = np.histogram(detector_data, bins = binning, density = True)

    return detector_data

def global_best_dists(model, data, sample_position, TRUE_FEATURES, iteration, save_path, FLAG):
    binning = np.arange(0, 150, 1)
    plt.figure()
    if FLAG == "TIMES":
        plt.hist(model, bins = binning, density = True, label = f"t1: {round(sample_position[0],3)} | t2: {round(sample_position[1], 3)} | t3: {round(sample_position[2], 3)} | t4: {round(sample_position[3], 3)}", histtype = "step")
        plt.hist(data, bins = binning, density = True, label = f"Detector Data | t1: {TRUE_FEATURES[0]} | t2: {TRUE_FEATURES[1]} | t3: {TRUE_FEATURES[2]} | t4: {TRUE_FEATURES[3]}", histtype = "step")
    if FLAG == "AMPLITUDES":
        plt.hist(model, bins = binning, density = True, label = f"A1: {round(sample_position[4],3)} | A2: {round(sample_position[5], 3)} | A3: {round(sample_position[6], 3)} | A4: {round(sample_position[7], 3)}", histtype = "step")
        plt.hist(data, bins = binning, density = True, label = f"Detector Data | A1: {TRUE_FEATURES[4]} | A2: {TRUE_FEATURES[5]} | A3: {TRUE_FEATURES[6]} | A4: {TRUE_FEATURES[7]}", histtype = "step")
    # plt.step(binning, model, label = f"t1: {round(sample_position[0],3)} | t2: {round(sample_position[1], 3)} | A1: {round(sample_position[2], 3)} | A2: {round(sample_position[3], 3)}")
    # plt.step(binning, data, label = f"Detector Data | t1: {TRUE_FEATURES[0]} | t2: {TRUE_FEATURES[1]} | A1: {TRUE_FEATURES[2]} | A2: {TRUE_FEATURES[3]}")
    plt.legend()
    plt.ylim((0,0.2))
    plt.title(f"MC vs Data Agreement | Iteration {iteration}")
    plt.xlabel("Time Residual (ns)")
    plt.ylabel("Normalised Counts per 1 ns Bin")
    plt.savefig(f"{save_path}/{iteration}.png")
    plt.close()

# monte-carlo sampling settings
NUM_SAMPLES = 20000
# inputs to PDF we want to recover
T1_TRUE = 5
T2_TRUE = 15
T3_TRUE = 150
T4_TRUE = 350
A1_TRUE = 0.7
A2_TRUE = 0.2
A3_TRUE = 0.07
A4_TRUE = 0.03
TRUE_FEATURES = [T1_TRUE, T2_TRUE, T3_TRUE, T4_TRUE, A1_TRUE, A2_TRUE, A3_TRUE, A4_TRUE]

# create the "True" dataset to be obtained by the optimiser
DATA = create_dataset(NUM_SAMPLES, T1_TRUE, T2_TRUE, T3_TRUE, T4_TRUE, A1_TRUE, A2_TRUE, A3_TRUE, A4_TRUE)

# optimiser settings
FLAG = "TIMES"      # either optimise the times OR the amplitudes
SAVE_PATH = "frames/4d_residuals"
ITERATIONS = 20
SCALE = [1, 0.5]                # RBF kernel amplitude and feature scale factors
EPSILON = 0.5                   # exploration / convergence factor for acquisition function
COST  = np.zeros(ITERATIONS)  # measured chi2 difference from sampling true function with given feature vector
NUM_FEATURES = 4                # number of features to consider
GLOBAL_BEST = 1e6               # initial value of global best cost 
GLOBAL_BEST_VALS = []           # array tracks successive global bests over time 
GLOBAL_BEST_FEATURES = np.zeros((ITERATIONS, NUM_FEATURES))
CONVERGENCE_COUNT = 0           # NEED TO CHECK THE CONVERGENCE LOGIC
T1_RES = 1                      # step size of feature 1 domain
T2_RES = 1   
T3_RES = 20
T4_RES = 50                   
A1_RES = 0.1
A2_RES = 0.2
A3_RES = 0.2
A4_RES = 0.2
T1_DOMAIN = [1, 10]             # domain of feature 1
T2_DOMAIN = [10, 20]
T3_DOMAIN = [100, 200]
T4_DOMAIN = [300, 500]
A1_DOMAIN = [0.01, 1]
A2_DOMAIN = [0.01, 1]
A3_DOMAIN = [0.01, 1]
A4_DOMAIN = [0.01, 1]
T1_AXIS = np.arange(T1_DOMAIN[0], T1_DOMAIN[1]+T1_RES, T1_RES) # discretised domain of feature 1
T2_AXIS = np.arange(T2_DOMAIN[0], T2_DOMAIN[1]+T2_RES, T2_RES)
T3_AXIS = np.arange(T3_DOMAIN[0], T3_DOMAIN[1]+T3_RES, T3_RES) 
T4_AXIS = np.arange(T4_DOMAIN[0], T4_DOMAIN[1]+T4_RES, T4_RES) 
A1_AXIS = np.arange(A1_DOMAIN[0], A1_DOMAIN[1]+A1_RES, A1_RES) 
A2_AXIS = np.arange(A2_DOMAIN[0], A2_DOMAIN[1]+A2_RES, A2_RES)
A3_AXIS = np.arange(A3_DOMAIN[0], A3_DOMAIN[1]+A3_RES, A3_RES) 
A4_AXIS = np.arange(A4_DOMAIN[0], A4_DOMAIN[1]+A4_RES, A4_RES) 
TIME_PER_ITER = [] # track performance as time for each loop to deploy

# set up of optimizer arrays  
measured_pts = np.zeros((ITERATIONS, NUM_FEATURES), dtype = np.float16) # (num_measurements, num_features)

# predicted points more complicated setup --> need every possible combination of feature vector points 
predicted_array_start = time.time()
if FLAG == "TIMES":
    # set the value of the amplitudes to use whilst optimising the time constants
    AMPLITUDES = [0.7, 0.2, 0.07, 0.03]#[0.5, 0.5, 0.5, 0.5]  # update these once better amplitudes are found !
    predicted_pts = np.zeros((len(T1_AXIS), len(T2_AXIS), len(T3_AXIS), len(T4_AXIS), NUM_FEATURES), dtype =np.float16)
    axes_collection = [T1_AXIS, T2_AXIS, T3_AXIS, T4_AXIS]
elif FLAG == "AMPLITUDES":
    # as above
    TIMES = [1, 10, 100, 450]#[5, 10, 300, 500] # update once better values for the times are found!
    predicted_pts = np.zeros((len(A1_AXIS), len(A2_AXIS), len(A3_AXIS), len(A4_AXIS), NUM_FEATURES), dtype =np.float16)
    axes_collection = [A1_AXIS, A2_AXIS, A3_AXIS, A4_AXIS]
else:
    print("invalid flag.")
    quit
print(axes_collection[0])
for (i, val1) in enumerate(axes_collection[0]):
    print(val1)
    for (j, val2) in enumerate(axes_collection[1]):
        for (k, val3) in enumerate(axes_collection[2]):
            for (l, val4) in enumerate(axes_collection[3]):
                predicted_pts[i,j,k,l,0] = val1
                predicted_pts[i,j,k,l,1] = val2
                predicted_pts[i,j,k,l,2] = val3
                predicted_pts[i,j,k,l,3] = val4
print(f"Took {time.time() - predicted_array_start} s to create predictions.")
# reshape so we have (num_predicted_points, num_features) 2D array
predicted_pts = predicted_pts.reshape(len(axes_collection[0]) * len(axes_collection[1]) \
                * len(axes_collection[2]) * len(axes_collection[3]), NUM_FEATURES)

last = None
for iteration in range(ITERATIONS):
    iter_loop_start = time.time()
    print(f"########### iteration {iteration} ###########")
    
    # mean and covariance of each predicted point
    mu_predicted, cov_predicted = gaussian_process_4d(measured_pts, COST, predicted_pts, SCALE)
    std_predicted = np.sqrt(abs(np.diag(cov_predicted)))

    # given mean, covariance and global best solution, return the expected improvement function at each predicted point
    expectedImprovement = acquistion_EXPECTED_IMPROVEMENT(mu_predicted, std_predicted, GLOBAL_BEST, EPSILON)
    max_improvement_idx = np.argmax(expectedImprovement)
    
    # maximimise the expected improvement and select it as the next iterations sampling point
    sample_position = predicted_pts[max_improvement_idx, :]      # each row in predicted_pts is a possible sample point
    print(f"Best improvement at pred point: {sample_position}")

    # checking convergence of sampling: if same point chosen many times to sample, exit optimisation
    if np.array_equal(sample_position, last):
        convergence_count +=1
    else:
        # reset convergence counter when different sample position chosen
        convergence_count = 0
    last = sample_position
    if convergence_count == 5:
        print(f"Converged in {iteration} iterations.\nGlobal Best Position: {GLOBAL_BEST_FEATURES}")
        break
    
    # add the chosen sampled point to the measurements array
    measured_pts[iteration, 0] = sample_position[0]
    measured_pts[iteration, 1] = sample_position[1]
    measured_pts[iteration, 2] = sample_position[2]
    measured_pts[iteration, 3] = sample_position[3]

    # for the amplitudes --> normalise them so they sum to 1
    if FLAG == "AMPLITUDES":
        magnitude = np.sum(sample_position)
        sample_position /= magnitude
    
        # use chosen position to generate a fake "monte carlo" simulation
        # need to add together the correct arrays in the sample position (ie including the previous AMPS if tuning the TIMES)
        sample_position = TIMES + sample_position.tolist()
    if FLAG == "TIMES":
        sample_position = sample_position.tolist() + AMPLITUDES
        print(type(AMPLITUDES))
        print(type(sample_position))
    model = create_dataset(NUM_SAMPLES, *sample_position)
    binning = np.arange(0, 150, 1)
    model_counts, _ = np.histogram(model, bins = binning, density = True)
    data_counts, _  = np.histogram(DATA, bins = binning, density = True)

    # evaluate cost function by histogramming model and finding sum of squared diff between bins
    goodness_of_fit = np.sum((model_counts - data_counts)**2)

    # compare goodness of fit to current global best: if better solution found, save model/data histograms
    if goodness_of_fit < GLOBAL_BEST:
        # update global best value
        GLOBAL_BEST = goodness_of_fit
        if FLAG == "TIMES":
            GLOBAL_BEST_FEATURES[iteration, :] = sample_position[:4]
        if FLAG == "AMPLITUDES":
            GLOBAL_BEST_FEATURES[iteration, :] = sample_position[4:]
        print(f"NEW GLOBAL BEST SOLUTION: {sample_position}")

        # create plot of true data vs updated best model
        global_best_dists(model, DATA, sample_position, TRUE_FEATURES, iteration, SAVE_PATH, FLAG)
    
    # update iteration by iteration tracker of global best and cost function
    GLOBAL_BEST_VALS.append(GLOBAL_BEST)
    COST[iteration] = goodness_of_fit

    TIME_PER_ITER.append(time.time() - iter_loop_start)

# display plot of global best over time (to see how many iterations before no further improvement)
plt.figure()
plt.plot(np.arange(0, iteration+1, 1), GLOBAL_BEST_VALS)
if FLAG == "TIMES":
    plt.title("Global Best " + r"$\chi ^2$" + " vs Iterations | Timing |")
if FLAG == "AMPLITUDES":
    plt.title("Global Best " + r"$\chi ^2$" + " vs Iterations | Amplitudes |")
plt.xlabel("Iteration")
plt.ylabel("Global Best " + r"$\chi ^2$")
plt.yscale("log")

plt.figure()
plt.plot(np.arange(0, iteration+1, 1), TIME_PER_ITER)
plt.title("Execution time per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Time (s)")
plt.plot([], [], label = f"Average time per Iter: {round(np.mean(TIME_PER_ITER), 3)} s")
plt.legend()
plt.show()