from point_selector import PointSelector
from plot_utils import *
import matplotlib
matplotlib.use("Agg")
import numpy as np
import json
import string
from copy import deepcopy
"""
Script uses PointSelector class object or random to select next set of parameters
to sample at.
"""

def create_macro(decay_consts, amplitudes, rise_time, name):
    """
    For each round of simulations, we load in a template macro and fill in the
    parameters to sample at. This macro is then used by each of the simulation
    jobs, with a different output file specified in the executable.
    """

    with open("/home/hunt-stokes/bayesian_optimisation/bi214_template.mac", "r") as infile:
        raw_text = string.Template(infile.read())
    
    output_text = raw_text.substitute(MATERIAL = "labppo_2p2_scintillator", T1 = decay_consts[0], T2 = decay_consts[1],
                                      T3 = decay_consts[2], T4 = decay_consts[3], TR = rise_time, A1 = amplitudes[0],
                                      A2 = amplitudes[1], A3 = amplitudes[2], A4 = amplitudes[3])

    with open(f"macros/{name}.mac", "w") as outfile:
        outfile.write(output_text)
    
    # dynamically create the submit file for simulations to point to the correct macro file
    
    with open("/home/hunt-stokes/bayesian_optimisation/submit_files/simulate.submit", "w") as outfile:
        outfile.write("""
        executable = /home/hunt-stokes/bayesian_optimisation/executables/submit_simulations.sh
        arguments  = $(Process) {MAC_NAME}.mac
        log        = /home/hunt-stokes/bayesian_optimisation/condor_logs/sim_logs/$(Process).log
        output     = /home/hunt-stokes/bayesian_optimisation/condor_logs/sim_outputs/$(Process).out
        error      = /home/hunt-stokes/bayesian_optimisation/condor_logs/sim_errors/$(Process).err
        request_memory = 1024MB
        queue 10
        """.format(MAC_NAME = f"{name}"))

with open("/home/hunt-stokes/bayesian_optimisation/algo_log.txt", "a") as outlog:
    outlog.write("\n\n")
    outlog.write("### THIS IS SELECT PARAMETERS SCRIPT ###\n")
    print("\n\n")
    print("### THIS IS SELECT PARAMETERS SCRIPT ###\n")
    # load up the current state of the algorithm
    with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "r") as f:
        info = json.load(f)

    # find what parameters we are optimising and what iteration we are in
    sample_info     = info['iteration_info']['current_block']['param_sampling']
    iteration       = sample_info['param_sample_iter']
    curr_params     = sample_info['current_parameters']

    # define the parameter names which match idx from curr_params
    parameter_names =  np.array(["T1", "T2", "T3", "T4", "TR", "A1", "A2", "A3", "A4"])

    # define the domains of each tunable parameter
    granuality = 50 # number of points in each parameter space
    t1         = np.linspace(1, 14, granuality)
    t2         = np.linspace(10, 90, granuality)
    t3         = np.linspace(60, 150, granuality)
    t4         = np.linspace(200, 500, granuality)
    tr         = np.linspace(0.1, 2.0, granuality)

    # define the length scales for the kernel
    l1         = np.linspace(0.5, 10, 50)   # do not make the minimum values too small or it blows the kernel up!
    l2         = np.linspace(2, 100, 50)
    l3         = np.linspace(10, 30, 50)
    l4         = np.linspace(50, 100, 50)
    ltheta     = np.linspace(0.1, 2, 20)  
    length_scales = np.array([l1, l2, l3, l4, l1, ltheta, ltheta, ltheta, ltheta], dtype = object)

    # depending on what amplitudes are being tuned, we have different max weights
    if curr_params == [5, 6]:
        max_weight = 0.9
        weights    = np.linspace(0.01, max_weight, granuality)
    if curr_params == [7, 8]:
        max_weight = 0.1
        weights    = np.linspace(0.01, max_weight, granuality)

    """
    A few different cases possible for iteration 0:

    For DECAY CONSTANT tuning:

        1. First time this parameter optimisation is run in any case (algo_iter 1, block_iter 1)
            --> randomly initialise parameters

        2. A block has restarted (algo iter N, block iter > 1). 
            --> We have a measured value from the end of the last block iter
            --> create new measured_pts array with single row = best solution from previous block iter
            --> select next point with PointSelector object

        3. Entire algorithm has restarted (algo iter > 1, block_iter = 1)
            --> We have a measured value from the end of the last algorithm
            --> create new measured_pts array with single row = best solution from previous algorithm
            --> select next point with PointSelector object

    For AMPLITUDE tuning:
        1. ALWAYS have a previously measured point from the DECAY CONSTANT tuning.
            --> create new measured_pts array with single row = best solution from DECAY CONSTANT tuning
            --> select next point with PointSelector object

    For RISE TIME : goes the same way as the AMPLITUDES.

    if iteration > 0, we load the measured_pts array for this algo_block_params and use Pointselector object.
    """

    algo_iter  = info['iteration_info']['full_algo_iter']
    block_iter = info['iteration_info']['current_block']['iteration']
    block_name = info['iteration_info']['current_block']['block_name']
    block_best = info['iteration_info']['current_block']['block_best_params']
    outlog.write(f"\n SELECT PARAMETERS for ITERATION {iteration} in BLOCK {block_name} iter {block_iter} and ALGO LOOP {algo_iter}.")
    print(f"\n SELECT PARAMETERS for ITERATION {iteration} in BLOCK {block_name} iter {block_iter} and ALGO LOOP {algo_iter}.")
    # deal with amplitudes and rise times
    if curr_params == [5, 6] or curr_params == [7, 8] or curr_params == [4]:

        feature_name   = parameter_names[curr_params[0]]
        plot_name    = f"{feature_name}_ALGO_{algo_iter}_BLOCK_{block_iter}_{iteration}" 
        if curr_params[0] == 4:
            feature_domain = tr
        else:
            feature_domain = weights
        outlog.write(f"\nTuning {feature_name}.")
        print(f"\nTuning {feature_name}.")
        
        # specify the previously sampled points
        if iteration == 0:
            # grab previous measured best solution from decay constant tuning
            
            best_decay_measurement = [block_best[f'{parameter_names[curr_params[0]]}'], block_best['obj']]
            outlog.write(f"\n\n## First iteration of amplitude / rise time tuning. Starting values loaded from previously measured points: {best_decay_measurement}. ##")
            print(f"\n\n## First iteration of amplitude / rise time tuning. Starting values loaded from previously measured points: {best_decay_measurement}. ##")
            # create new measured points array with this sample
            measured_points = np.array([best_decay_measurement]) # single row
        else:
            # already created a measured points array with everything in - load it up
            measured_points = np.load(f"measured_points/{feature_name}_ALGO_{algo_iter}_BLOCK_{block_iter}.npy")
            # measured_points = measured_points.reshape((len(measured_points), 1))
        print(measured_points, measured_points.shape)
        # set up PointSelector Object to choose next location to sample --> 1D optimisation
        Optimiser                = PointSelector()
        Optimiser.name           = feature_name
        Optimiser.iteration      = iteration
        Optimiser.measured_pts   = measured_points[:, 0].reshape((len(measured_points), 1))
        Optimiser.measured_vals  = measured_points[:, 1]
        Optimiser.feature_domain = [granuality]
        Optimiser.predicted_pts  = feature_domain.reshape((granuality, 1))
        Optimiser.length_scales  = length_scales[curr_params[0]]

        # use the previously measured point to update surrogate, uncertainty and acquisition function
        Optimiser.update_surrogate()
        next_sample      = Optimiser.lower_confidence_bound()
        updated_features = feature_domain[next_sample[0]]
        outlog.write(f"\nSelected {feature_name} = {updated_features} as next sample position.")
        print(f"\nSelected {feature_name} = {updated_features} as next sample position.")
        # update the measured points array
        measured_points = measured_points.tolist()
        measured_points.append([updated_features, 10000]) # no measurement yet
        np.save(f"measured_points/{feature_name}_ALGO_{algo_iter}_BLOCK_{block_iter}.npy", measured_points)

        # obtain the mean, uncertainty and acquisition function for plotting this iteration result
        mu_pred  = Optimiser.mean_func
        cov_pred = Optimiser.cov_func
        acq_pred = Optimiser.acq_func_eval
        surrogate_uncert_acquistion_1d(mu_pred, cov_pred, acq_pred, Optimiser.predicted_pts, plot_name, iteration, measured_points)

        # create macro used to simulate all this
        parameters = info['parameters']
        decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
        amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
        rise_time       = [parameters["TR"]]

        params_update   = np.array(decay_constants + rise_time + amplitudes)
        if feature_name == "TR":
            params_update[curr_params] = updated_features
        else:
            params_update[curr_params] = [updated_features, max_weight - updated_features]
        create_macro(params_update[0:4], params_update[5:], params_update[4], feature_name)

        # check if we trigger a convergence point update
        last_measured   = parameters[f'{feature_name}']
        perc_difference = abs(last_measured - updated_features) / last_measured
        outlog.write(f"\n% Change from last measurement is: {last_measured} --> {updated_features} ({perc_difference}) %.")
        print(f"\n% Change from last measurement is: {last_measured} --> {updated_features} ({perc_difference}) %.")
        if perc_difference <= 0.05:
            conv_points = info['iteration_info']['current_block']['param_sampling']['conv_points']
            outlog.write(f"Convergence criteria met! Incremementing convergence criteria from {conv_points} to {conv_points+1}.")
            print(f"Convergence criteria met! Incremementing convergence criteria from {conv_points} to {conv_points+1}.")
            info['iteration_info']['current_block']['param_sampling']['conv_points'] += 1
        else:
            # reset convergence counter
            outlog.write(f"\nResetting convergence counter to 0.")
            print(f"\nResetting convergence counter to 0.")
            info['iteration_info']['current_block']['param_sampling']['conv_points'] = 0

        # update parameters in the log file
        info['parameters'][f'{feature_name}'] = updated_features
        if feature_name != "TR":
            info['parameters'][f'{parameter_names[curr_params[1]]}'] = max_weight - updated_features
            
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as config_file:
            json.dump(info, config_file, indent = 4)

    else:
        """
        We perform a 2D optimisation over decay constants. For iteration 0, 
        there are a range of possible starts.
        """
        curr_params   = np.array(curr_params)
        feature_names = parameter_names[curr_params]
        plot_name    = f"{feature_names[0]}_{feature_names[1]}_ALGO_{algo_iter}_BLOCK_{block_iter}_{iteration}" 
        if algo_iter == 0 and block_iter == 0 and iteration == 0 and np.all(curr_params == np.array([0, 1])):
            # this is truly the first time these decay constants have been tuned - random initialisation
            random_sample_idx = np.random.randint(granuality, size = 2) # randomly pick index to sample
            updated_features  = np.array([t1[random_sample_idx[0]], t2[random_sample_idx[1]]])
            outlog.write(f"\nFirst iteration for algorithm. Randomly selected {feature_names[0]} = {updated_features[0]} and {feature_names[1]} = {updated_features[1]}.")
            print(f"\nFirst iteration for algorithm. Randomly selected {feature_names[0]} = {updated_features[0]} and {feature_names[1]} = {updated_features[1]}.")
            # create macro
            parameters = info['parameters']
            decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
            amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
            rise_time       = [parameters["TR"]]
            params_update   = np.array(decay_constants + rise_time + amplitudes)
            params_update[curr_params] = updated_features
            create_macro(params_update[0:4], params_update[5:], params_update[4], f"{feature_names[0]}_{feature_names[1]}")

            # update the parameters in JSON log
            info['parameters']['T1'] = updated_features[0]
            info['parameters']['T2'] = updated_features[1]
            
            # update the algo_start parameters
            algo_start = deepcopy(info['iteration_info']['initial_parameters']) # deepcopy because nested dicts mutable
            info['iteration_info']['initial_parameters']['T1'] = updated_features[0]
            info['iteration_info']['initial_parameters']['T2'] = updated_features[1]
            outlog.write(f"\nUpdated ALGO_START params from {algo_start} --> {info['iteration_info']['initial_parameters']}.")
            print(f"\nUpdated ALGO_START params from {algo_start} --> {info['iteration_info']['initial_parameters']}.")
            # update the block_start parameters
            block_start = deepcopy(info['iteration_info']['current_block']['prev_params'])
            info['iteration_info']['current_block']['prev_params']['T1'] = updated_features[0]
            info['iteration_info']['current_block']['prev_params']['T2'] = updated_features[1]
            outlog.write(f"\nUpdated BLOCK_START params from {block_start} to {info['iteration_info']['current_block']['prev_params']}.")
            print(f"\nUpdated BLOCK_START params from {block_start} to {info['iteration_info']['current_block']['prev_params']}.")
            # create a measured_points array for the future iterations
            measured_points = [updated_features.tolist() + [1000]]
            np.save(f"measured_points/{feature_names[0]}_{feature_names[1]}_ALGO_{algo_iter}_BLOCK_{block_iter}.npy", measured_points)

        # there exists a previously measured point for a given (t1, t2) or (t3, t4) --> stored in parameters in JSON file
        else:
            parameters      = info['parameters']
            if iteration == 0:
                outlog.write(f"\n\nStarting selection for iteration {iteration} of {feature_names} as result of block / algo restart.")
                print(f"\n\nStarting selection for iteration {iteration} of {feature_names} as result of block / algo restart.")
                # load the previously measured points array
                best_decay_measurement = [block_best[f'{parameter_names[curr_params[0]]}'], block_best[f'{parameter_names[curr_params[1]]}'], block_best['obj']]
                measured_points        = np.array([best_decay_measurement])
                outlog.write(f"\nLoaded previously measured points with values: {measured_points}.")
                print(f"\nLoaded previously measured points with values: {measured_points}.")
            else:
                # not iteration zero so load up the measured points array that already exists as a np array
                measured_points = np.load(f"measured_points/{feature_names[0]}_{feature_names[1]}_ALGO_{algo_iter}_BLOCK_{block_iter}.npy")
            print(measured_points)
            # regardless of the source of the measured_points array, the process is the same
            if curr_params[0] == 0:
                feature_domain = [t1, t2]
            else:
                feature_domain = [t3, t4]
            # create the predicted points array
            predicted_points = np.zeros((granuality**2, 2))
            pt_counter       = 0
            for i in range(granuality):
                for j in range(granuality):
                    predicted_points[pt_counter, 0] = feature_domain[0][i]
                    predicted_points[pt_counter, 1] = feature_domain[1][j]
                    pt_counter += 1
            
            # create the Optimiser and define attributes
            Optimiser                = PointSelector()
            Optimiser.name           = feature_names
            Optimiser.iteration      = iteration
            Optimiser.measured_pts   = measured_points[:, 0:2].reshape((len(measured_points), 2))
            Optimiser.measured_vals  = measured_points[:, 2]
            Optimiser.feature_domain = [granuality, granuality]
            Optimiser.predicted_pts  = predicted_points
            Optimiser.length_scales  = np.array([length_scales[curr_params[0]], length_scales[curr_params[1]]])

            # use to optimiser to select the next points to sample at
            Optimiser.update_surrogate()
            next_sample = Optimiser.lower_confidence_bound()
            updated_features = np.array([feature_domain[0][next_sample[0]], feature_domain[1][next_sample[1]]])
            outlog.write(f"\nNext sample at {feature_names[0]} = {updated_features[0]} | {feature_names[1]} = {updated_features[1]}.")
            print(f"\nNext sample at {feature_names[0]} = {updated_features[0]} | {feature_names[1]} = {updated_features[1]}.")
            # update and save the measured pts array for this sampled position
            measured_points = measured_points.tolist()
            measured_points.append([feature_domain[0][next_sample[0]], feature_domain[1][next_sample[1]], 10000])
            np.save(f"/home/hunt-stokes/bayesian_optimisation/measured_points/{feature_names[0]}_{feature_names[1]}_ALGO_{algo_iter}_BLOCK_{block_iter}.npy", measured_points)

            # create the plots showing output from this iteration
            mu_pred      = Optimiser.mean_func 
            cov_pred     = Optimiser.cov_func
            acq_pred     = Optimiser.acq_func_eval
            meshX, meshY = np.meshgrid(feature_domain[0], feature_domain[1])
            surrogate_uncert_acquistion(mu_pred, cov_pred, acq_pred, meshX, meshY, plot_name, iteration, measured_points)

            # create the macro that simulates this
            decay_constants = [parameters["T1"], parameters["T2"], parameters["T3"], parameters["T4"]]
            amplitudes      = [parameters["A1"], parameters["A2"], parameters["A3"], parameters["A4"]]
            rise_time       = [parameters["TR"]]
            params_update = np.array(decay_constants + rise_time + amplitudes)
            params_update[curr_params] = updated_features
            create_macro(params_update[0:4], params_update[5:], params_update[4], f"{feature_names[0]}_{feature_names[1]}")

            # check for convergence criteria being triggered
            last_measured   = np.array([parameters[f'{feature_names[0]}'], parameters[f'{feature_names[1]}']])
            perc_difference = abs(last_measured - updated_features) / last_measured
            outlog.write(f"\n% Change from last measurement is: {last_measured} --> {updated_features} ({perc_difference}) %.")
            print(f"\n% Change from last measurement is: {last_measured} --> {updated_features} ({perc_difference}) %.")
            if np.all(perc_difference <= 0.05):
                conv_points = info['iteration_info']['current_block']['param_sampling']['conv_points']
                outlog.write(f"Convergence criteria met! Incremementing convergence criteria from {conv_points} to {conv_points+1}.")
                print(f"Convergence criteria met! Incremementing convergence criteria from {conv_points} to {conv_points+1}.")
                info['iteration_info']['current_block']['param_sampling']['conv_points'] += 1
            else:
                # reset convergence counter
                outlog.write(f"\nResetting convergence counter to 0.")
                print(f"\nResetting convergence counter to 0.")
                info['iteration_info']['current_block']['param_sampling']['conv_points'] = 0


            # update the parameter defaults in JSON log with latest measured point
            info['parameters'][f'{feature_names[0]}'] = updated_features[0]
            info['parameters'][f'{feature_names[1]}'] = updated_features[1]
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as config_file:
            json.dump(info, config_file, indent = 4)