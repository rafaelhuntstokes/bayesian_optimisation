import json
import sys

"""
Script called whenever a block of [ constants --> amplitudes ] finishes. Determine
whether to resubmit the entire block to iteratively home-in on best constants and
amplitudes.
"""

def move_to_next_block():
    """
    Function moves the algorithm to begin next parameter tuning block. This occurs when: 

    1) block finishes and solution has converged
    2) block finishes and reached max_iters
    """

    # update the default parameters with this block's best result
    best_params        = info['iteration_info']['current_block']['block_best_params']
    info['parameters'] = best_params
    # reset the block tracking data to the next block in the sequence
    info['iteration_info']['current_block']['block_name'] = "SECOND_PAIR" if block_name == "FIRST_PAIR" else "RISE_TIME"
    info['iteration_info']['current_block']['iteration']  = 0
    info['iteration_info']['current_block']['param_sampling']['param_sample_iter'] = 0
    info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [2, 3] if block_name == "FIRST_PAIR" else [4]

    # for the rise time we don't need to iterate the block submission, so set max iters to 1 and 10 otherwise
    #info['iteration_info']['current_block']['max_iter'] = 3 #if block_name != "RISE_TIME" else 1
    outlog.write(f"\nUpdated block name to {info['iteration_info']['current_block']['block_name']}. \
                 Tuning parameters: {info['iteration_info']['current_block']['param_sampling']['current_parameters']}.")
    # update JSON log file
    with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
        json.dump(info, outfile, indent = 4)

# open logfile to write all decisions / outputs to
with open("/home/hunt-stokes/bayesian_optimisation/algo_log.txt", "a") as outlog:
    outlog.write("\n\n")
    outlog.write("### THIS IS BLOCK TERMINATE SCRIPT ###\n")
    # gather necessary information from the JSON file
    with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "r") as f:
        info = json.load(f)
    block_info   = info['iteration_info']['current_block']
    iteration    = block_info['iteration']
    max_iters    = block_info['max_iter']
    start_params = block_info['prev_params']
    end_params   = block_info['block_best_params']
    block_name   = block_info['block_name']
    outlog.write(f"\nCompleted block {block_name} interation {iteration}.")

    # calculate the % difference in parameters before and after this block
    perc_changes = [abs(start_params[f'{key}'] - end_params[f'{key}']) / start_params[f'{key}'] for key in start_params.keys() if start_params[f'{key}'] > 0]
    outlog.write(f"\nPercentage changes between start and end of block are: \n{perc_changes}")
    keys         = list(start_params.keys())

    resubmit_flag = False
    for perc in range(len(perc_changes)):
        if perc_changes[perc] > 0.05:
            # we have not converged for all parameters
            resubmit_flag = True
            outlog.write(f"\nParameter {f'{keys[perc]}'} did not converge")
            break
    if resubmit_flag == False:
        outlog.write(f"\nBlock {block_name} has converged within tolerance. The final parameters are: \n {end_params}")
        # update the default parameters with this block's converged result
        if block_name != "RISE_TIME":
            move_to_next_block()
        else:
            outlog.write("\n\n ### COMPLETED RISE_TIME BLOCK. ###")
            outlog.write(f"\nUpdating block_start to block_best and exiting to algo terminate script.")
            outlog.write(f"\nUpdated block start from {info['iteration_info']['current_block']['prev_params']} --> {end_params}.")
            info['iteration_info']['current_block']['prev_params']['T1'] = end_params['T1']
            info['iteration_info']['current_block']['prev_params']['T2'] = end_params['T2']
            info['iteration_info']['current_block']['prev_params']['T3'] = end_params['T3']
            info['iteration_info']['current_block']['prev_params']['T4'] = end_params['T4']
            info['iteration_info']['current_block']['prev_params']['TR'] = end_params['TR']
            info['iteration_info']['current_block']['prev_params']['A1'] = end_params['A1']
            info['iteration_info']['current_block']['prev_params']['A2'] = end_params['A2']
            info['iteration_info']['current_block']['prev_params']['A3'] = end_params['A3']
            info['iteration_info']['current_block']['prev_params']['A4'] = end_params['A4']
            with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
                json.dump(info, outfile, indent = 4)
        # converged so exit
        sys.exit(0) # next node
    
    elif block_name == "RISE_TIME":
        # we are at the end of the RISE_TIME block. Need to update block_start to block_best and exit.
        outlog.write("\n\n ### COMPLETED RISE_TIME BLOCK. ###")
        outlog.write(f"\nUpdating block_start to block_best and exiting to algo terminate script.")
        outlog.write(f"\nUpdated block start from {info['iteration_info']['current_block']['prev_params']} --> {end_params}.")
        info['iteration_info']['current_block']['prev_params'] = end_params
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
            json.dump(info, outfile, indent = 4)
        sys.exit(0) # next node
    
    elif iteration < max_iters:
        # unconverged and less than max iters
        outlog.write("\n\n############################################################")
        outlog.write(f"\n\nResubmitting block {block_name} iteration {iteration+1} of {max_iters}.")
        outlog.write("\n############################################################")

        # refresh starting parameters and iteration number
        iteration    = iteration + 1
        outlog.write(f"\n\nSetting BLOCK_START to BLOCK_BEST\n{info['iteration_info']['current_block']['prev_params']} --> {end_params}.")
        # set the previous block params to the best measured performance of this block iteration
        info['iteration_info']['current_block']['iteration']         = iteration
        info['iteration_info']['current_block']['prev_params']['T1'] = end_params['T1']
        info['iteration_info']['current_block']['prev_params']['T2'] = end_params['T2']
        info['iteration_info']['current_block']['prev_params']['T3'] = end_params['T3']
        info['iteration_info']['current_block']['prev_params']['T4'] = end_params['T4']
        info['iteration_info']['current_block']['prev_params']['TR'] = end_params['TR']
        info['iteration_info']['current_block']['prev_params']['A1'] = end_params['A1']
        info['iteration_info']['current_block']['prev_params']['A2'] = end_params['A2']
        info['iteration_info']['current_block']['prev_params']['A3'] = end_params['A3']
        info['iteration_info']['current_block']['prev_params']['A4'] = end_params['A4']
        
        # move starting parameters back to the start of the block
        if block_name == "FIRST_PAIR":
            info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [0, 1]
        else:
            info['iteration_info']['current_block']['param_sampling']['current_parameters'] = [2, 3]

        # reset the iterations of the param_sampling and convergence points
        info['iteration_info']['current_block']['param_sampling']['param_sample_iter'] = 0
        info['iteration_info']['current_block']['param_sampling']['conv_points']       = 0
        outlog.write(f"\nReset param sample iters and conv points to 0.")
        outlog.write(f"\nUpdated tuning parameters to {info['iteration_info']['current_block']['param_sampling']['current_parameters']}.")
        # update JSON log file
        with open("/home/hunt-stokes/bayesian_optimisation/opto_log.JSON", "w") as outfile:
            json.dump(info, outfile, indent = 4)
        sys.exit(1) # resubmit
    
    else:
        
        # unconverged but current iteration = max_iters - take best found block solution forwards
        move_to_next_block()
        outlog.write(f"\nBlock {block_name} has reached max_iters: moving to next block.\nSolution did NOT converge.")
        sys.exit(0) # next node