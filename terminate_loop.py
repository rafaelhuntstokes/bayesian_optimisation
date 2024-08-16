import json
import sys
import os
"""
Script called at the end of each optimisation loop to see whether the optimiser will continue
or move to the next node.
"""
# open logfile
with open("opto_log.JSON", "r") as logfile:
    log = json.load(logfile)

# work out if we are at the final iteration
with open("/home/hunt-stokes/bayesian_optimisation/post_log.txt", "w") as logfile:
    if log["terminate_flag"] == True:
        logfile.write("\nchi2 has converged - exiting.")
        logfile.write(f"\nUse LAST MEASURED POINT: {log['last_measured']}")
        sys.exit(100) # custom exit code that instructs the entire dag to exit
    elif log["current_iteration"] < log["max_iteration"]:
        # restart the loop
        logfile.write("\nBeginning next loop.")
        log["current_iteration"] = log["current_iteration"] + 1
        with open("opto_log.JSON", "w") as outlog:
            json.dump(log, outlog, indent = 4)
        sys.exit(1)
    else:
        logfile.write("\nMax iters reached. Beginning next node . . .")
        
        # need to update the idx of the parameters over which to tune
        current_parameters = log["current_parameters"]
        if current_parameters[-1] == 7:
            # we enter the rise time tuning regime
            log["current_parameters"] = [8]
        elif current_parameters[-1] == 8:
            logfile.write("\nWe have finished running the optimisation. Creating final plots and exiting ...")
            os.system("python3 select_parameters.py")
            sys.exit(0)
        else:
            log["current_parameters"] = [current_parameters[0]+1, current_parameters[1]+1]

        logfile.write(f"\nUpdating default simulation parameters to: {log['global_best']['parameters']}")
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

        # reset current iteration
        log["current_iteration"] = 0
        # reset the GBest parameters
        # log["global_best"]["objective"] = 1e9
        # log["global_best"]["parameters"]["T1"] = 999
        # log["global_best"]["parameters"]["T2"] = 999
        # log["global_best"]["parameters"]["T3"] = 999
        # log["global_best"]["parameters"]["T4"] = 999
        # log["global_best"]["parameters"]["TR"] = 999
        # log["global_best"]["parameters"]["A1"] = 999
        # log["global_best"]["parameters"]["A2"] = 999
        # log["global_best"]["parameters"]["A3"] = 999
        # log["global_best"]["parameters"]["A4"] = 999
        # update json log file so next set of parameters are chosen for optimising over
        with open("opto_log.JSON", "w") as outlog:
            json.dump(log, outlog, indent = 4)
        sys.exit(0)