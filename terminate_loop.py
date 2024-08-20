import json
import sys

"""
Script called at the end of each optimisation loop to see whether the optimiser will continue
or move to the next node.
"""

def update_optimisation_parameters(log, converged_flag):
    """
    Function updates the current tuning parameters to the next in the sequence, 
    or exits if we have finished tuning TR.
    """

    current_parameters = log["current_parameters"]
    if current_parameters[-1] == 7:
        # we enter the rise time tuning regime
        log["current_parameters"] = [8]
        logfile.write(f"\nMax iters reached. Beginning next optimisation of Rise Time (param idx {log["current_parameters"]}).")
    elif current_parameters[-1] == 8:
        logfile.write(f"\nWe have finished optimising all parameters. The best solution was: \n{log["global_best"]}.\nExiting!")
        sys.exit(0)
    else:
        logfile.write(f"\nMax iters reached. Beginning next optimisation of param idxs: {log["current_parameters"]}).")
        log["current_parameters"] = [current_parameters[0]+1, current_parameters[1]+1]

    if converged_flag == False:
        logfile.write(f"\nNo convergence. Updating default simulation parameters to:\n{log['global_best']['parameters']}.")
    
        # update the default simulation parameters with GBest parameters
        log["parameters"]["T1"] = log["global_best"]["parameters"]["T1"]
        log["parameters"]["T2"] = log["global_best"]["parameters"]["T2"]
        log["parameters"]["T3"] = log["global_best"]["parameters"]["T3"]
        log["parameters"]["T4"] = log["global_best"]["parameters"]["T4"]
        log["parameters"]["TR"] = log["global_best"]["parameters"]["TR"]
        log["parameters"]["A1"] = log["global_best"]["parameters"]["A1"]
        log["parameters"]["A2"] = log["global_best"]["parameters"]["A2"]
        log["parameters"]["A3"] = log["global_best"]["parameters"]["A3"]
        log["parameters"]["A4"] = log["global_best"]["parameters"]["A4"]
    else:
        logfile.write(f"\nConverged! Updating default simulation parameters previous sample:\n{log['last_measured']}.")
    
    # reset current iteration
    log["current_iteration"] = 0
    
    # update json log file so next set of parameters are chosen for optimising over
    with open("opto_log.JSON", "w") as outlog:
        json.dump(log, outlog, indent = 4)

# open logfile
with open("opto_log.JSON", "r") as logfile:
    log = json.load(logfile)

# work out if we are at the final iteration - write output to logfile.
with open("/home/hunt-stokes/bayesian_optimisation/post_log.txt", "a") as logfile:

    logfile.write(f"\nWe have reached the end of iteration {log["current_iteration"]} for parameter(s) {log["current_parameters"]}.")
    if log["terminate_flag"] == True:
        logfile.write(f"\nKS-test PASSED. Use last measured point as tuning: \
                      Time Constants: {log["last_measured"]["T1"]}, {log["last_measured"]["T2"]}, {log["last_measured"]["T3"]}, {log["last_measured"]["T4"]}\n \
                      Amplitudes: {log["last_measured"]["A1"]}, {log["last_measured"]["A2"]}, {log["last_measured"]["A3"]}, {log["last_measured"]["A4"]}\n \
                      Rise Time: {log["last_measured"]["TR"]}.")
        sys.exit(100) # custom exit code that instructs the entire dag to exit
    
    if log["current_iteration"] < log["max_iteration"]:
        # restart the loop - unless we have reached the max convergence counts!
        current_parameters = log["current_parameters"]
        names = ["T1", "T2", "T3", "T4"]
        if log["convergence_flags"]["conv_points"] == 5:
            if current_parameters[0] == 8:
                parameter_name = "TR"
            else:
                parameter_name = names[current_parameters[0]]
            # we have converged on a solution for this stage so exit and begin new stage
            logfile.write(f"\nThe optimisation has CONVERGED. Moving to next parameter optimisation.")
            log["convergence_flags"]["conv_points"] = 0
            log["convergence_flags"][f"{parameter_name}"] = True

            # replace the default parameters with the converged solutuion
            # update the default simulation parameters with GBest parameters
            log["parameters"]["T1"] = log["last_measured"]["T1"]
            log["parameters"]["T2"] = log["last_measured"]["T2"]
            log["parameters"]["T3"] = log["last_measured"]["T3"]
            log["parameters"]["T4"] = log["last_measured"]["T4"]
            log["parameters"]["TR"] = log["last_measured"]["TR"]
            log["parameters"]["A1"] = log["last_measured"]["A1"]
            log["parameters"]["A2"] = log["last_measured"]["A2"]
            log["parameters"]["A3"] = log["last_measured"]["A3"]
            log["parameters"]["A4"] = log["last_measured"]["A4"]

            # we move to the next parameter to tune
            update_optimisation_parameters(log, True)
            sys.exit(0) # begin next node

        else:
            # not converged and not reached max iters so begin next loop!
            log["current_iteration"] = log["current_iteration"] + 1
            logfile.write(f"\nBeginning next loop as iteration: (iteration {log["current_iteration"]}).")
        
            with open("opto_log.JSON", "w") as outlog:
                json.dump(log, outlog, indent = 4)
            sys.exit(1)
    else:
        # need to update the idx of the parameters over which to tune
        update_optimisation_parameters(log, False)
        sys.exit(0) # begin next node