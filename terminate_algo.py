import json
import sys

"""
Script is called whenever a full cycle of tuning is completed. Terminate called
if change in every parameter is less than 5 %.
"""

# open the output logfile ready to write results to
with open("/home/hunt-stokes/bayesian_optimisation/algo_log.txt", "a") as outlog:
    outlog.write("\n\n")
    outlog.write("### THIS IS ALGO TERMINATE SCRIPT ###\n")
    # gather neccessary information from JSON file
    with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "r") as f:
        info = json.load(f)

    start_params = info['iteration_info']['initial_parameters']
    end_params   = info['iteration_info']['current_block']['block_best_params']
    iteration    = info['iteration_info']['full_algo_iter']
    max_iters    = info['iteration_info']['max_iter']
    outlog.write(f"\nCompleted entire algo interation {iteration}.")

    # do not have access to numpy in post-scripts (don't ask me why)
    perc_changes = [abs(start_params[f'{key}'] - end_params[f'{key}']) / start_params[f'{key}'] for key in start_params.keys() if start_params[f'{key}'] > 0]
    outlog.write(f"\nPercentage changes between start and end of algorithm are: \n{perc_changes}")
    keys         = list(start_params.keys())

    resubmit_flag = False
    for perc in range(len(perc_changes)):
        if perc_changes[perc] > 0.05:
            # we have not converged for all parameters
            resubmit_flag = True
            outlog.write(f"\nParameter {f'{keys[perc]}'} did not converge. Restarting algo!")
            break
    if resubmit_flag == False:
        # converged so exit
        outlog.write(f"\nAll parameters have converged within tolerance. The final parameters are: \n {end_params}")
        sys.exit(0)
    elif iteration < max_iters:
        # unconverged and less than max iters
        outlog.write(f"\nSubmitting full algo iteration {iteration+1} of {max_iters}.")
        outlog.write("\n ############################################################")

        # refresh starting parameters and iteration number
        info['iteration_info']['initial_parameters']['T1'] = end_params['T1']
        info['iteration_info']['initial_parameters']['T2'] = end_params['T2']
        info['iteration_info']['initial_parameters']['T3'] = end_params['T3']
        info['iteration_info']['initial_parameters']['T4'] = end_params['T4']
        info['iteration_info']['initial_parameters']['TR'] = end_params['TR']
        info['iteration_info']['initial_parameters']['A1'] = end_params['A1']
        info['iteration_info']['initial_parameters']['A2'] = end_params['A2']
        info['iteration_info']['initial_parameters']['A3'] = end_params['A3']
        info['iteration_info']['initial_parameters']['A4'] = end_params['A4']
        info['iteration_info']['full_algo_iter']     = iteration + 1

        info['iteration_info']['current_block']['block_name'] = "FIRST_PAIR"
        info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [0, 1]
        # info['iteration_info']['current_block']['max_iter'] = 5
        # update JSON log file
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
            json.dump(info, outfile, indent = 4)
        sys.exit(1) # resubmit
    else:
        # unconverged but current iteration = max_iters - simply raise warning and exit
        outlog.write(f"\nAlgorithm has reached max_iters: exiting.\nSolution did NOT converge.\nBest params: \n{end_params}")
        sys.exit(0) # exit