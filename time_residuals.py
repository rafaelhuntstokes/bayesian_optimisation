import numpy as np
import rat
from ROOT import RAT
from scipy.stats import chisquare, chi2, ks_2samp
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from plot_utils import time_residual_agreement

def extract_residuals():
    """
    Calculates the time residuals from a given round of simulation .root files.
    """

    time_residuals = []
    counter        = 0 
    for ientry, _ in rat.dsreader("/data/snoplus3/hunt-stokes/bayesian_optimisation_sims/*.root"):

        # setup time residual calculator and point3d classes to handle AV offset
        PMTCalStatus = RAT.DU.Utility.Get().GetPMTCalStatus()
        light_path = rat.utility().GetLightPathCalculator()
        group_velocity = rat.utility().GetGroupVelocity()
        pmt_info = rat.utility().GetPMTInfo()
        psup_system_id = RAT.DU.Point3D.GetSystemId("innerPMT")
        av_system_id = RAT.DU.Point3D.GetSystemId("av")

        if ientry.GetEVCount() == 0:
            continue

        reconEvent = ientry.GetEV(0)

        # check reconstruction is valid
        fit_name = reconEvent.GetDefaultFitName()
        if not reconEvent.FitResultExists(fit_name):
            continue

        vertex = reconEvent.GetFitResult(fit_name).GetVertex(0)
        if (not vertex.ContainsPosition() or
            not vertex.ContainsTime() or
            not vertex.ValidPosition() or
            not vertex.ValidTime() or
            not vertex.ContainsEnergy() or
            not vertex.ValidEnergy()):
            continue
        # print("Reconstruction checks PASSED!")
        # reconstruction valid so get reconstructed position and energy
        reconPosition  = vertex.GetPosition() # returns in PSUP coordinates
        reconEnergy    = vertex.GetEnergy()        
        reconEventTime = vertex.GetTime()
        
        # apply AV offset to position
        event_point = RAT.DU.Point3D(psup_system_id, reconPosition)
        event_point.SetCoordinateSystem(av_system_id)
        if event_point.Mag() > 4000:
            continue
        # convert back to PSUP coordinates
        event_point.SetCoordinateSystem(psup_system_id)

        # apply energy tagging cuts the same as that in data
        if reconEnergy < 1.25 or reconEnergy > 3.00:
            continue

        # event has passed all the cuts so we can extract the time residuals
        calibratedPMTs = reconEvent.GetCalPMTs()
        pmtCalStatus = rat.utility().GetPMTCalStatus()
        for j in range(calibratedPMTs.GetCount()):
            pmt = calibratedPMTs.GetPMT(j)
            if pmtCalStatus.GetHitStatus(pmt) != 0:
                continue
            
            # residual_recon = timeResCalc.CalcTimeResidual(pmt, reconPosition, reconEventTime, True)
            pmt_point = RAT.DU.Point3D(psup_system_id, pmt_info.GetPosition(pmt.GetID()))
            light_path.CalcByPosition(event_point, pmt_point)
            inner_av_distance = light_path.GetDistInInnerAV()
            av_distance = light_path.GetDistInAV()
            water_distance = light_path.GetDistInWater()
            transit_time = group_velocity.CalcByDistance(inner_av_distance, av_distance, water_distance)
            residual_recon = pmt.GetTime() - transit_time - reconEventTime
            
            time_residuals.append(residual_recon)
        
        counter += 1
        if counter % 100 == 0:
            print("COMPLETED {} / {}".format(counter, 5000))

    return time_residuals

def ks_test(data, model):
    ks_statistic, p_value = ks_2samp(data, model)

    print(f"Performed K-S test. Statistic: {ks_statistic} | p-value: {p_value}")

    return ks_statistic, p_value

# load the JSON log
with open("opto_log.JSON", "r") as infile:
    log = json.load(infile)
feature_names  = np.array(["T1", "T2", "T3", "T4", "theta1", "theta2", "theta3", "theta4", "TR"])
current_params = np.array(log["current_parameters"])
if len(current_params) == 2:
    print(current_params)
    feature_name = feature_names[current_params] 
    dof_params   = 2
    measured_pts = f"measured_points/{feature_name[0]}_{feature_name[1]}.npy"
else:
    feature_name = feature_names[current_params[0]]
    dof_params   = 1
    measured_pts = f"measured_points/{feature_name}.npy"

# calculate the time residuals from this iteration's simulations
time_residuals = extract_residuals()
print("Extracted simulated residuals.")

# load up the data residuals to compare to
data_residuals = np.load("/data/snoplus3/hunt-stokes/tune_cleaning/detector_data/bi_FV4000.0_goldList.npy", allow_pickle = True)
data_residuals = np.concatenate(data_residuals)

# create the binned histograms and calculate the chi2
binning           = np.arange(-5, 250, 1)
binned_sim, edges = np.histogram(time_residuals, bins = binning, density = True)
binned_data, _    = np.histogram(data_residuals, bins = binning, density = True)

# remove bins with zero counts from chi2 calculation
keep_idx    = np.nonzero(binned_sim)
binned_sim  = binned_sim[keep_idx]
binned_data = binned_data[keep_idx]

# chi2 is unreliable with small bin counts so we scale everything up to a large number
binned_sim  = binned_sim  * 1e7
binned_data = binned_data * 1e7
print(f"Minimum bin count in data and MC are: {np.amin(binned_data)}, {np.amin(binned_sim)}")

# calculate the chi2 for this iteration
dof   = len(keep_idx[0]) - 1 - dof_params
chi2_stat, p_value = chisquare(f_obs = binned_data, f_exp = binned_sim) 
print(dof, chi2_stat)

ks_stat, ks_p_val = ks_test(data_residuals, time_residuals)

# check if the chi2 statistic < global best:
# if chi2_stat < log["global_best"]["objective"]:
if ks_stat < log["global_best"]["objective"]:
    # print(f"Updating global best chi2 from {log['global_best']['objective']} to {chi2_stat}.")
    print(f"Updating global best ks from {log['global_best']['objective']} to {ks_stat}.")
    # log["global_best"]["objective"] = chi2_stat
    log["global_best"]["objective"] = ks_stat

    # update the global best parameters
    log["global_best"]["parameters"]["T1"] = log["last_measured"]["T1"]
    log["global_best"]["parameters"]["T2"] = log["last_measured"]["T2"]
    log["global_best"]["parameters"]["T3"] = log["last_measured"]["T3"]
    log["global_best"]["parameters"]["T4"] = log["last_measured"]["T4"]
    log["global_best"]["parameters"]["TR"] = log["last_measured"]["TR"]
    log["global_best"]["parameters"]["A1"] = log["last_measured"]["A1"]
    log["global_best"]["parameters"]["A2"] = log["last_measured"]["A2"]
    log["global_best"]["parameters"]["A3"] = log["last_measured"]["A3"]
    log["global_best"]["parameters"]["A4"] = log["last_measured"]["A4"]

    with open("opto_log.JSON", "w") as outfile:
        json.dump(log, outfile, indent = 4)

# need to adjust the p-value since we optimised over 2 parameters
adjusted_p_value = chi2.sf(chi2_stat, dof)

# check if we may accept the current model as matching the data (hypothesis test)
# print(f"The p-value is: {adjusted_p_value}")
print(f"The p-value is: {ks_p_val}")

# if adjusted_p_value > 0.05:
if ks_p_val > 0.05:
    print("Model matches the data! Ending optimisation...")
    log["terminate_flag"] = True

    with open("opto_log.JSON", "w") as outfile:
        json.dump(log, outfile, indent = 4)
else:
    print("Model does not match data within tolerance. Continuing optimisation...")

# regardless of what happens, let's make the output plots
time_residual_agreement(data_residuals, time_residuals, f"{log['current_parameters']}_{log['current_iteration']}")

# need to update the measured points and values
# first check if the measured points array exists
print(feature_name)
print(f"/home/hunt-stokes/bayesian_optimisation/{measured_pts}")
if os.path.isfile(f"/home/hunt-stokes/bayesian_optimisation/{measured_pts}") == True:
    # we can load it as it already exists
    measured_vals   = np.load(f"/home/hunt-stokes/bayesian_optimisation/{measured_pts}")
    
    # only need to set the objective value of the last measured point
    # measured_vals[-1, 2] = chi2_stat
    print("Updating objective value to: ", ks_stat)

    if len(current_params) == 1:
        measured_vals[-1, 1] = ks_stat
    else:
        measured_vals[-1, 2] = ks_stat
    print(measured_vals, measured_vals.shape)
    np.save(f"/home/hunt-stokes/bayesian_optimisation/{measured_pts}", measured_vals)
else:
    # this is probably the first iteration so need to create it
    # measured_vals = [chi2_stat]
    measured_vals = [ks_stat]
    np.save("/home/hunt-stokes/bayesian_optimisation/measured_vals.npy", measured_vals)