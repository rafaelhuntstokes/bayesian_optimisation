from point_selector import PointSelector
from plot_utils import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import string
import os

"""
Script creates a PointSelector class object that chooses the next position to
sample the objective function at.
"""

def create_macro(decay_consts, amplitudes, rise_time, material, name):
    """
    For each round of simulations, we load in a template macro and fill in the
    parameters to sample at. This macro is then used by each of the simulation
    jobs, with a different output file specified in the executable.
    """

    with open("bi214_template.mac", "r") as infile:
        raw_text = string.Template(infile.read())
    
    output_text = raw_text.substitute(MATERIAL = material, T1 = decay_consts[0], T2 = decay_consts[1],
                                      T3 = decay_consts[2], T4 = decay_consts[3], TR = rise_time, A1 = amplitudes[0],
                                      A2 = amplitudes[1], A3 = amplitudes[2], A4 = amplitudes[3])

    with open(f"macros/{name}.mac", "w") as outfile:
        outfile.write(output_text)
    
    # dynamically create the submit file for simulations to point to the correct macro file
    
    with open("submit_files/simulate.submit", "w") as outfile:
        outfile.write("""
        executable = executables/submit_simulations.sh
        arguments  = $(Process) {MAC_NAME}.mac
        log        = condor_logs/sim_logs/$(Process).log
        output     = condor_logs/sim_outputs/$(Process).out
        error      = condor_logs/sim_errors/$(Process).err

        queue 50
        """.format(MAC_NAME = f"{name}"))
        
def softmax_normalisation(vec_theta):
    """
    Function uses softmax function to convert the optimisation theta variables --> Amplitude
    weights used in the MC.
    """

    return np.exp(vec_theta) / np.sum(np.exp(vec_theta)) 

def sum_normalisation(vec_weights, keep_idx):
    """
    Normalise each of the input weights by the sum of the weights.
    """

    # optimiser constant value
    const = vec_weights[keep_idx]
    # find the amount the non-optimised weights need to contribute to the whole
    residual = 1 - const

    # work out a scale factor for the remaining weights
    scaling  = residual / (np.sum(vec_weights) - const)

    print(f"The optimised parameter is: {const}.\nThe residual is therefore: {residual}.\nThe scale factor is: {scaling}\n.")

    # normalise and reset
    vec_weights = vec_weights * scaling
    vec_weights[keep_idx] = const
    
    print(f"Sanity check! The weights are: {vec_weights} and they sum to: {np.sum(vec_weights)}.")

    return vec_weights
    

# define the feature space for each parameter
size     = 50 # using the same granuality for each feature - probably not great
material = "labppo_2p2_scintillator"
t1       = np.linspace(1, 6, size)
t2       = np.linspace(10, 50, size)
t3       = np.linspace(60, 150, size)
t4       = np.linspace(200, 500, size)
tr       = np.linspace(0.1, 2.0, size)
theta    = np.linspace(0.01, 1.0, size)

# also need to define the length scales over which to tune the constants for each parameter
l1     = np.linspace(0.01, 5, 50)
l2     = np.linspace(2, 10, 50)
l3     = np.linspace(10, 30, 50)
l4     = np.linspace(50, 100, 50)
ltheta = np.linspace(0.01, 2, 20)
length_scales = np.array([l1, l2, l3, l4, ltheta, ltheta, ltheta, ltheta, l1])
feature_spaces = np.array([t1, t2, t3, t4, theta, theta, theta, theta, tr])
feature_names  = ["T1", "T2", "T3", "T4", "theta1", "theta2", "theta3", "theta4", "TR"]

# load up the JSON log file to find the current iteration
log = open("opto_log.JSON")
log = json.load(log)

iteration = log["current_iteration"]
print(f"\n###### Iteration {iteration} #####\n")

# find which parameters are being tuned
current_parameters = log["current_parameters"]
if len(current_parameters) == 2:
    feature1 = feature_spaces[current_parameters[0]]
    feature2 = feature_spaces[current_parameters[1]]
    feature_name = f"{feature_names[current_parameters[0]]}_{feature_names[current_parameters[1]]}"
else:
    feature1 = feature_spaces[current_parameters]
    feature_name = f"{feature_names[current_parameters[0]]}"
    
print("Optimising parameter(s): ", feature_name)
if iteration == 0:
    # we randomly select a point from the feature space
    if len(current_parameters) == 2:
        next_sample_idx = np.random.randint(size, size = 2)
        print(f"Sampling at: {feature_spaces[current_parameters[0]][next_sample_idx[0]], feature_spaces[current_parameters[1]][next_sample_idx[1]]}")
        updated_feature1 = feature_spaces[current_parameters[0]][next_sample_idx[0]]
        updated_feature2 = feature_spaces[current_parameters[1]][next_sample_idx[1]]
        updated_features  = np.array([updated_feature1, updated_feature2]) 
    else:
        next_sample_idx = np.random.randint(size, size = 1)
        print(f"Sampling at: {feature_spaces[current_parameters][0][next_sample_idx]}")
        updated_features = np.array(feature_spaces[current_parameters][0][next_sample_idx])
    
    # create the macro used to run all these jobs
    parameters = log["parameters"]
    decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
    amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
    rise_time       = [parameters["TR"]]

    # update the values
    params_update = np.array(decay_constants + amplitudes + rise_time)
    params_update[current_parameters] = updated_features

    # given the current status of the theta parameters, apply softmax to return normalised weights
    # updated_weights    = softmax_normalisation(params_update[4:8])

    # only update the weights if we are in the 2 parameter tuning regime - otherwise just tuning tR
    if len(current_parameters) == 2:
        updated_weights    = sum_normalisation(params_update[4:8], current_parameters[1]-4)
        params_update[4:8] = updated_weights

    # update the parameters in the JSON log
    log["last_measured"]["T1"] = params_update[0]
    log["last_measured"]["T2"] = params_update[1]
    log["last_measured"]["T3"] = params_update[2]
    log["last_measured"]["T4"] = params_update[3]
    log["last_measured"]["A1"] = params_update[4]
    log["last_measured"]["A2"] = params_update[5]
    log["last_measured"]["A3"] = params_update[6]
    log["last_measured"]["A4"] = params_update[7]
    log["last_measured"]["TR"] = params_update[8]

    with open("opto_log.JSON", "w") as config_file:
        json.dump(log, config_file, indent = 4)

    # update the parameters in question with the next sample point
    create_macro(params_update[0:4], params_update[4:8], params_update[8], material, feature_name)

    # create the measured vals array for subsequent use
    measured_vals = [updated_features.tolist() + [1000]]
    print("iter 0 measured vals array: ", measured_vals)
    np.save(f"measured_points/{feature_name}.npy", measured_vals)
    
else:
    # we use the PointSelector Bayesian approach
    Optimiser = PointSelector()

    # load the previously measured points array and objective function vals
    measured_pts  = np.load(f"measured_points/{feature_name}.npy") # 2D array of measured points for given set of params
    
    if len(current_parameters) == 2:
        # setup the predicted points array
        predicted_pts = np.zeros((size**2, 2)) # predicted point for every combination of feature values
        pt_counter    = 0
        feature1_axis = feature_spaces[current_parameters[0]]
        feature2_axis = feature_spaces[current_parameters[1]]
        for i in range(size):
            for j in range(size):
                predicted_pts[pt_counter, 0] = feature1_axis[i]
                predicted_pts[pt_counter, 1] = feature2_axis[j]
                pt_counter += 1
            
        # set the feature domains and measured points for the Optimiser to use
        Optimiser.name           = feature_name
        Optimiser.iteration      = iteration 
        Optimiser.measured_pts   = measured_pts[:, 0:2]
        Optimiser.measured_vals  = measured_pts[:, 2]

        Optimiser.feature_domain = [size, size] # simply specifies the dimensions of the feature space
        Optimiser.predicted_pts  = predicted_pts

        # select the appropriate kernel parameters to optimise over
        Optimiser.length_scales  = np.array([length_scales[current_parameters[0]], length_scales[current_parameters[1]]])

        # use previosuly measured points to update surrogate and covariance
        Optimiser.update_surrogate()

        # use these updated values to select the next point to simulate at
        next_sample = Optimiser.lower_confidence_bound()
        print(next_sample)
        updated_features = [feature1_axis[next_sample[0]], feature2_axis[next_sample[1]]]
        print(f"Sampling at: {feature1_axis[next_sample[0]]}, {feature2_axis[next_sample[1]]}")

        # update and save the measured points array
        measured_pts = measured_pts.tolist()
        measured_pts.append([feature1_axis[next_sample[0]], feature2_axis[next_sample[1]], 10000]) # no measurement yet
        np.save(f"measured_points/{feature_name}.npy", measured_pts)

        # obtain the 2D arrays for the plots
        mu_pred  = Optimiser.mean_func
        cov_pred = Optimiser.cov_func 
        acq_pred = Optimiser.acq_func_eval
        meshX, meshY = np.meshgrid(feature1_axis, feature2_axis)
        
        # create the plots
        surrogate_uncert_acquistion(mu_pred, cov_pred, acq_pred, meshX, meshY, feature_name, iteration, measured_pts)

        # create the macro used to run all these jobs
        parameters = log["parameters"]
        decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
        amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
        rise_time       = [parameters["TR"]]

        # update the values
        params_update = np.array(decay_constants + amplitudes + rise_time)
        params_update[current_parameters] = updated_features

        # given the current status of the theta parameters, apply softmax to return normalised weights
        # updated_weights    = softmax_normalisation(params_update[4:8])

        
        updated_weights    = sum_normalisation(params_update[4:8], current_parameters[1]-4)
        params_update[4:8] = updated_weights

        # update the parameters in the JSON log
        log["last_measured"]["T1"] = params_update[0]
        log["last_measured"]["T2"] = params_update[1]
        log["last_measured"]["T3"] = params_update[2]
        log["last_measured"]["T4"] = params_update[3]
        log["last_measured"]["A1"] = params_update[4]
        log["last_measured"]["A2"] = params_update[5]
        log["last_measured"]["A3"] = params_update[6]
        log["last_measured"]["A4"] = params_update[7]
        log["last_measured"]["TR"] = params_update[8]

        with open("opto_log.JSON", "w") as config_file:
            json.dump(log, config_file, indent = 4)

        # update the parameters in question with the next sample point
        create_macro(params_update[0:4], params_update[4:8], params_update[8], material, feature_name)
    else:
        # only 1 D optimisation but otherwise it's the same process
        # setup the predicted points array        
        print(current_parameters)
        feature1_axis = feature_spaces[current_parameters[0]]
        predicted_pts = feature1_axis.reshape((size, 1))       # simply a 1D axis
            
        # set the feature domains and measured points for the Optimiser to use
        Optimiser.name           = feature_name
        Optimiser.iteration      = iteration
        Optimiser.measured_pts   = measured_pts[:, 0].reshape((len(measured_pts), 1))
        Optimiser.measured_vals  = measured_pts[:, 1]
        Optimiser.feature_domain = [size]
        Optimiser.predicted_pts  = predicted_pts
        Optimiser.length_scales  = length_scales[current_parameters[0]]
        
        # use previosuly measured points to update surrogate and covariance
        Optimiser.update_surrogate()
        next_sample      = Optimiser.lower_confidence_bound()
        updated_features = feature1_axis[next_sample[0]] 
        print(next_sample)
        print(f"Sampling at: {feature1_axis[next_sample[0]]}")

        # update the measured points array
        measured_pts = measured_pts.tolist()
        measured_pts.append([feature1_axis[next_sample[0]], 10000]) # no measurement yet
        np.save(f"measured_points/{feature_name}.npy", measured_pts)

        # obtain the 2D arrays for the plots
        mu_pred  = Optimiser.mean_func
        cov_pred = Optimiser.cov_func 
        acq_pred = Optimiser.acq_func_eval
        print(cov_pred)
        # create the plots
        surrogate_uncert_acquistion_1d(mu_pred, cov_pred, acq_pred, predicted_pts, feature_name, iteration, measured_pts)
        
        # create the macro used to run all these jobs
        parameters      = log["parameters"]
        decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
        amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
        rise_time       = [parameters["TR"]]

        # update the values
        params_update = np.array(decay_constants + amplitudes + rise_time)
        params_update[current_parameters] = updated_features

        # update the parameters in the JSON log
        log["last_measured"]["T1"] = params_update[0]
        log["last_measured"]["T2"] = params_update[1]
        log["last_measured"]["T3"] = params_update[2]
        log["last_measured"]["T4"] = params_update[3]
        log["last_measured"]["A1"] = params_update[4]
        log["last_measured"]["A2"] = params_update[5]
        log["last_measured"]["A3"] = params_update[6]
        log["last_measured"]["A4"] = params_update[7]
        log["last_measured"]["TR"] = params_update[8]

        with open("opto_log.JSON", "w") as config_file:
            json.dump(log, config_file, indent = 4)

        # update the parameters in question with the next sample point
        create_macro(params_update[0:4], params_update[4:8], params_update[8], material, feature_name)