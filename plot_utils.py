import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
import json

"""
Series of plotting functions to create intermediate plots of the surrogates, 
uncertainties, marginalisation plots for each variable over time, acquisition
functions and correlaton matrices.

Also shows the global best value found for each parameter optimisation pairing over time.
"""

def surrogate_uncert_acquistion(mean, uncertainty, acquisition, meshX, meshY, name, iteration, measured_pts):
    """
    Function creates a panel of plots showing 3D and 2D contour plots of the surrogate,
    uncertainty and acquisition functions.
    """

    # handle when only 1 measured point and array is 1D for imshow heatmap plots
    measured_pts = np.array(measured_pts)
    if measured_pts.size == 3:
        measured_pts = measured_pts[None, :]
    else:
        print(measured_pts)

    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 12))
    for i in range(3):
        axes[0, i].remove()  # Remove the existing 2D subplot placeholder
        axes[0, i] = fig.add_subplot(2, 3, i+1, projection='3d')

    axes[0,0].plot_surface(meshX, meshY, mean, cmap = "inferno")
    axes[0,0].set_title("Surrogate")
    axes[0,0].set_xlabel(f"{name[0:2]}")
    axes[0,0].set_ylabel(f"{name[3:5]}")
    axes[0,0].set_zlabel("Surrogate")

    axes[0,1].plot_surface(meshX, meshY, uncertainty, cmap = "inferno")
    axes[0,1].set_title("Uncertainty")
    axes[0,1].set_xlabel(f"{name[0:2]}")
    axes[0,1].set_ylabel(f"{name[3:5]}")
    axes[0,1].set_zlabel("Uncertainty")

    axes[0,2].plot_surface(meshX, meshY, acquisition, cmap = "inferno")
    axes[0,2].set_title("Acquisition")
    axes[0,2].set_xlabel(f"{name[0:2]}")
    axes[0,2].set_ylabel(f"{name[3:5]}")
    axes[0,2].set_zlabel("acquisition")

    img     = axes[1,0].imshow(mean, origin = "lower", extent = [np.amin(meshY), np.amax(meshY), np.amin(meshX), np.amax(meshX)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,0])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,0].set_xlabel("Feature 2")
    axes[1,0].set_ylabel(f"{name[3:5]}")
    axes[1,0].scatter(measured_pts[:-1,1], measured_pts[:-1,0], color = "red", marker = "o")
    axes[1,0].plot(measured_pts[:,1], measured_pts[:,0], color = "red", linestyle = "--", marker = "")
    axes[1,0].scatter(measured_pts[-1,1], measured_pts[-1,0], color = "red", marker = "x")

    img     = axes[1,1].imshow(uncertainty, origin = "lower", extent = [np.amin(meshY), np.amax(meshY), np.amin(meshX), np.amax(meshX)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,1])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,1].set_xlabel(f"{name[3:5]}")
    axes[1,1].set_ylabel(f"{name[0:2]}")
    axes[1,1].scatter(measured_pts[:-1,1], measured_pts[:-1,0], color = "red", marker = "o")
    axes[1,1].plot(measured_pts[:,1], measured_pts[:,0], color = "red", linestyle = "--", marker = "")
    axes[1,1].scatter(measured_pts[-1,1], measured_pts[-1,0], color = "red", marker = "x")

    img     = axes[1,2].imshow(acquisition, origin = "lower", extent = [np.amin(meshY), np.amax(meshY), np.amin(meshX), np.amax(meshX)], aspect = "auto", cmap = "inferno")
    divider = make_axes_locatable(axes[1,2])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax = cax)
    axes[1,2].set_xlabel(f"{name[3:5]}")
    axes[1,2].set_ylabel(f"{name[0:2]}")

    plt.savefig(f"plots/{name}_{iteration}.pdf")
    plt.close()

def surrogate_uncert_acquistion_1d(mean, uncertainty, acquisition, axis, name, iteration, measured_pts):
    """
    Same as above but create a 1D plot version for the rise time tuning.
    """

    measured_pts = np.array(measured_pts)
    axis = axis
    mean = mean
    uncertainty = uncertainty
    acquisition = acquisition
    print(axis, axis.shape)
    fig, axes = plt.subplots(nrows = 1, ncols = 2)

    axes[0].plot(axis, mean, color = "black", linestyle = "--")
    axes[0].scatter(measured_pts[:-1,0], measured_pts[:-1,1], color = "black", marker = "o")
    axes[0].fill_between(axis[:,0], np.atleast_1d(mean-uncertainty), np.atleast_1d(mean+uncertainty), alpha = 0.6, color = "red")
    axes[0].set_xlabel(f"{name}")
    axes[0].set_ylabel("Surrogate")
    axes[1].plot(axis, acquisition, color = "black")
    axes[1].set_xlabel(f"{name}")
    axes[1].set_ylabel("Acquisition Function")
    fig.tight_layout()

    plt.savefig(f"plots/{name}_{iteration}.pdf")
    plt.close()

def time_residual_agreement(data, model, name):
    """
    Function makes a 3 subplot plot showing the binned tRes distributions between data and MC,
    alongside the empirical CDFs of each distribution and max difference between them which
    the ks-test returns.
    """

    def calculate_cdf(distro):
        """
        Returns the empirical CDF.
        """

        distro = np.sort(distro)

        cdf    = np.arange(1, len(distro)+1) / len(distro)

        return distro, cdf
    
    with open("opto_log.JSON", "r") as logfile:
        log = json.load(logfile)
    
    current_parameters = log["current_parameters"]
    # find the parameter values used
    parameter_names    = np.array(["T1", "T2", "T3", "T4", "A1", "A2", "A3", "A4", "TR"])
    names              = parameter_names[current_parameters]
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))
    binning = np.arange(-5, 350, 1)
    if len(names) == 2:
        values      = [log["last_measured"][names[0]], log["last_measured"][names[1]]]
        axes[0].plot([], [], linestyle = "", label = f"{names[0]}: {values[0]:.3f} ns | {names[1]} : {values[1]:.3f}")
        axes[1].plot([], [], linestyle = "", label = f"{names[0]}: {values[0]:.3f} ns | {names[1]} : {values[1]:.3f}")
    else:
        values      = log["last_measured"]["TR"]
        axes[0].plot([], [], linestyle = "", label = f"Rise Time: {values:.3f} ns")
        axes[1].plot([], [], linestyle = "", label = f"Rise Time: {values:.3f} ns")
    
    axes[0].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[0].hist(model, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC")
    
    axes[0].set_xlim((-5, 100))
    axes[0].set_xlabel("Time Residual [ns]")
    axes[0].legend()

    axes[1].hist(data, bins = binning, density = True, histtype = "step", color = "black", linewidth = 2, label = "data")
    axes[1].hist(model, bins = binning, density = True, histtype = "step", color = "red", linewidth = 2, label = "MC")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time Residual [ns]")
    axes[1].legend()

    x_measured, cdf_measured = calculate_cdf(data)
    x_simulated, cdf_simulated = calculate_cdf(model)

    # Create a common x-axis by combining the unique x-values from both distributions
    x_common = np.sort(np.unique(np.concatenate((x_measured, x_simulated))))

    # Interpolate the CDFs onto the common x-axis
    interp_cdf_measured = interp1d(x_measured, cdf_measured, bounds_error=False, fill_value=(0,1))
    interp_cdf_simulated = interp1d(x_simulated, cdf_simulated, bounds_error=False, fill_value=(0,1))

    cdf_measured_interp = interp_cdf_measured(x_common)
    cdf_simulated_interp = interp_cdf_simulated(x_common)

    # Calculate the KS statistic and find the position of the maximum difference
    ks_statistic = np.max(np.abs(cdf_measured_interp - cdf_simulated_interp))
    max_diff_index = np.argmax(np.abs(cdf_measured_interp - cdf_simulated_interp))

    # Plot CDFs
    axes[2].plot(x_common, cdf_measured_interp, label='Measured Data CDF')
    axes[2].plot(x_common, cdf_simulated_interp, label='Simulated Data CDF')
    axes[2].set_title(f'KS Test: Statistic = {ks_statistic:.3f}')

    # Highlight the maximum difference (KS statistic)
    axes[2].vlines(x_common[max_diff_index], cdf_measured_interp[max_diff_index], cdf_simulated_interp[max_diff_index], 
            colors='r', linestyle='--', label=f'Max Diff (KS statistic)')

    axes[2].set_xlabel('Data Value')
    axes[2].set_ylabel('CDF')
    axes[2].legend()
    
    fig.tight_layout()

    plt.savefig(f"plots/tres_comparison/{name}.png")
    plt.close()

def plot_ARD_LL(nlogml, params, length_scales, name, iteration):
    # create a plot of the likelihood and minimum point as a function of length scale parameters
    print(min(length_scales[1]), max(length_scales[1]), min(length_scales[0]), max(length_scales[0]))
    plt.imshow(nlogml, extent = [min(length_scales[1]), max(length_scales[1]), min(length_scales[0]), max(length_scales[0])], origin = "lower", aspect = "auto")
    plt.scatter(params[1], params[0], color = "red")
    plt.ylabel(f"{name[0:2]}")
    plt.xlabel(f"{name[3:5]}")
    plt.savefig(f"plots/ARD_{name}_{iteration}.png")
    plt.close()
def plot_ARD_LL_1d(nlogml, params, length_scale, name, iteration):
    plt.plot(length_scale, nlogml, color = "black")
    plt.xlabel(f"{name}")
    plt.ylabel(r"$log(\mathcal{L})$")
    plt.axvline(params, color = "red", linestyle = "--")
    plt.savefig(f"plots/ARD_{name}_{iteration}.png")
    plt.close()

