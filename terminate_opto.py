import json
import sys

"""
Script governs the resubmission or completion criteria for individual parameter 
optimisation loops. This occurs within the blocks.

It doesn't matter what parameters we are optimising, as long as we update the current
parameters and iterations upon exit.

Block termination will handle the rest.
"""

# open log file to track decisions
with open("/home/hunt-stokes/bayesian_optimisation/algo_log.txt", "a") as outlog:
    outlog.write("\n\n")
    outlog.write("### THIS IS OPTO TERMINATE SCRIPT ###\n")
    # gather necessary information from JSON file
    with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "r") as f:
        info = json.load(f)

    block_info   = info['iteration_info']['current_block']
    param_iter   = block_info['param_sampling']['param_sample_iter']
    max_iter     = block_info['param_sampling']['max_iter']
    conv_pts     = block_info['param_sampling']['conv_points']
    block_name   = block_info['block_name']
    start_params = block_info['param_sampling']['last_measured']
    curr_params  = block_info['param_sampling']['current_parameters']
    end_params   = info['parameters']
    outlog.write(f"\n\nCompleted iteration {param_iter} of block {block_name}.")

    # work out if we are within 5 % of the previous measurement
    perc_changes = [abs(start_params[f'{key}'] - end_params[f'{key}']) / start_params[f'{key}'] for key in start_params.keys() if start_params[f'{key}'] > 0]
    outlog.write(f"\nPercentage changes between start and end of param sample are: \n{perc_changes}")
    
    conv_flag = True
    for idx in curr_params:
        delta = perc_changes[idx]
        if delta > 0.05:
            conv_flag = False
    if conv_flag == True:
        conv_pts += 1
    
    if conv_pts == 5 or param_iter == max_iter:
        if conv_pts == 5:
            # we consider this has converged
            outlog.write(f"\nTuning has converged for parameters {curr_params} in block {block_name}.")
        else:
            outlog.write(f"\nMax iters reached for tuning parameters {curr_params} in block {block_name}.")
        # reset the iterations and conv points
        info['iteration_info']['current_block']['param_sampling']['conv_points']        = 0
        info['iteration_info']['current_block']['param_sampling']['param_sample_iter']  = 0
        outlog.write(f"\nReset convergence points and param sampling iteration to 0.")
        # move on to the next set of parameters within a block
        if block_name == "FIRST_PAIR" and curr_params == [0, 1]:
            info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [5, 6]
            outlog.write(f"\nUpdating tuning parameters to {info['iteration_info']['current_block']['param_sampling']['current_parameters']}.")
        elif block_name == "SECOND_PAIR" and curr_params == [2, 3]:
            info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [7, 8]
            outlog.write(f"Updating tuning parameters to {info['iteration_info']['current_block']['param_sampling']['current_parameters']}.")
        else:
            # we are either already tuning amplitudes, or we are tuning rise time --> converged or max iters here
            # may trigger a resubmitted block or a new block
            outlog.write(f"\nTuning amplitudes or rise time. Handing over to block terminate script...")
        
        # update JSON log file
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
            json.dump(info, outfile, indent = 4)
        sys.exit(0) # next node
    else:
        # not converged and not reached max-iters --> resubmit parameter sampling
        outlog.write(f"\nNo convergence or max iters. Proceeding to next iteration of parameters {curr_params} in block {block_name}.")
        info['iteration_info']['current_block']['param_sampling']['param_sample_iter']  += 1
        # update JSON log file
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
            json.dump(info, outfile, indent = 4)
        sys.exit(1) # resubmit node
        