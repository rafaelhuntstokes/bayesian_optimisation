import numpy as np
import matplotlib
matplotlib.use("Agg")
from plot_utils import *

"""
Script is the core of the Bayesian optimisation. Based on previously measured
values, update the mean and covariance function and select next point to sample.

Output plots of the current surrogate, acquisition and uncertainty.
"""

class PointSelector():

    def __init__(self):
        """
        Point selector needs to know the feature domain over which to evaluate
        the mean and variance functions, alongside the previously measured points
        and values.
        """

        # note! This class is agnostic to the dimensionality of the feature space
        self.feature_domain = None # (i, j) 2D vector containing the number of points in each feature axis
        self.predicted_pts  = None # (N, num_features) array for N predicted points
        self.measured_vals  = []   # (M, )  vector of objective value for each measured point
        self.measured_pts   = []   # (M, num_features) matrix of measured point positions 
        self.mean_func      = None # (N, N) array of mean func for each feature pair
        self.cov_func       = None # (N, N) array giving of covariance for each feature pair
        self.acq_func_eval  = None # (N, N) array giving the acquisition function for each feature pair
        self.hyperparam_obj = []
        self.length_scales  = None # (1, K) length scale parameters for K kernel parameters
        self.kernel_params  = None
        self.gradient_steps = 0.001  # how large to step when optimising kernel hyperparameters via gradient descent
        self.iteration      = None
        self.name           = None

        # keep the 2D covariance matrices available as attributes for inspection
        self.cov_pred       = None
        self.cov_meas       = None
        self.cov_meas_pred  = None

    def update_surrogate(self):
        """
        Function updates the mean and covariance function based on the measured values
        from previous iterations.

        MEASUREMENTS: (M,) vector of objective sample values of length M
        PREDICTIONS : (N, num_features) matrix of N predictions of num_features
        """

        # measured points and values are maintained as a list so convert to array
        self.measured_pts  = np.array(self.measured_pts)
        self.measured_vals = np.array(self.measured_vals)
        print("measured pts: ", self.measured_pts)
        print("measured_vals: ", self.measured_vals)
        
        # Automatic Relevance Determination (ARD) Step --> tune kernel hyperparameters until converged #

        # check if we have more than 1 measured point (otherwise don't do ARD tuning and set scales to mid points)
        if len(self.measured_pts[:,0]) > 1:
            print("Beginning ARD kernel tuning ...")
            self.tune_kernel()
        else:
            print("Only 1 measued point. Setting length scales to mid points of range.")
            if len(self.length_scales) == 2:
                # set kernel length scales to middle of each range
                axis1 = self.length_scales[0]
                axis2 = self.length_scales[1]
                l1    = axis1[len(axis1) // 2]
                l2    = axis2[len(axis2) // 2]
                self.kernel_params = np.array([l1, l2])
            else:
                self.kernel_params = np.array([self.length_scales[len(self.length_scales) // 2]])

        
        # covariance matrices between each set of points
        # to maintain numerical stability of inverse, small 'jitter' added to diagonal elements
        self.cov_pred      = self.kernel_rbf(self.predicted_pts, self.predicted_pts) + 1e-6*np.eye(len(self.predicted_pts))
        self.cov_meas      = self.kernel_rbf(self.measured_pts, self.measured_pts)   + 1e-6*np.eye(len(self.measured_pts))
        print("COV MEAS - PRED")
        self.cov_meas_pred = self.kernel_rbf(self.measured_pts, self.predicted_pts).T   # (N, N)

        print("####### CHECKING THE DIMENSIONS OF INTERMEDIATE MATRICES!\n")
        print(f"Shape cov_pred: {self.cov_pred.shape}")
        print(f"Shape of cov measured: {self.cov_meas.shape}")
        print(f"Shape of cov_meas_pred: {self.cov_meas_pred.shape}")

        # update the mean and uncertainty at each predicted point
        inv = np.linalg.inv(self.cov_meas)
        mu_predicted =  self.cov_meas_pred @ (inv @ self.measured_vals)
        cov_predicted = self.cov_pred - self.cov_meas_pred @ (inv @ self.cov_meas_pred.T)
        print(f"Shape of mean function: {mu_predicted.shape}")
        print(f"Shape of cov_predicted: {cov_predicted.shape}")
        print("####### #################################################\n")

        # shape to match the feature domain for plotting
        self.mean_func = mu_predicted.reshape(self.feature_domain)
        self.cov_func  = np.sqrt(np.abs(np.diag((cov_predicted)))).reshape(self.feature_domain)

        # recast the measured points and values back to list form
        self.measured_pts  = self.measured_pts.tolist()
        self.measured_vals = self.measured_vals.tolist()

    def tune_kernel(self):
        """
        Function uses ARD to tune the length scale hyperparameters in the RBF kernel.

        A simple grid search is employed.
        """

        def eval_log_marginal():
            """
            Function evaluates the -log(p) for a given set of length scales, l1 and l2.
            """
            
            rbf     = self.kernel_rbf(self.measured_pts, self.measured_pts)
            inv_rbf = np.linalg.inv(rbf)
            det_rbf = np.linalg.det(rbf)
            nlog_ml  = 0.5 * ( self.measured_vals.T @ inv_rbf @ self.measured_vals +  np.log(det_rbf) + len(self.measured_pts)*np.log(2*np.pi) )  
            return nlog_ml
        
        if len(self.length_scales) == 2:
            # if we have 2 features
            axis1 = self.length_scales[0]
            axis2 = self.length_scales[1]
            nlogml          = np.zeros((len(axis1), len(axis2)), dtype = np.float32)
            for i in range(len(axis1)):
                for j in range(len(axis2)):
                    
                    # find -logml for this set of length scales
                    l1 = axis1[i]
                    l2 = axis2[j]

                    # set the kernel parameters to use these length scales
                    self.kernel_params = np.array([l1, l2])

                    # evaluate log marginal likelihood
                    nlogml[i, j] = eval_log_marginal()

            # find the minimum in -logml to set the length scales for next iteration
            min_idx            = np.argwhere(nlogml == np.amin(nlogml))[0]
            print(f"Updated length scales to [{[axis1[min_idx[0]], axis2[min_idx[1]]]}]")
            self.kernel_params = np.array([axis1[min_idx[0]], axis2[min_idx[1]]])

            # create a plot of the likelihood and minimum point as a function of length scale parameters
            plot_ARD_LL(nlogml, self.kernel_params, self.length_scales, self.name, self.iteration)

        else:
            # we only have 1 feature scale to optimise over
            nlogml = np.zeros(len(self.length_scales), dtype = np.float32)

            for i in range(len(self.length_scales)):

                l1 = self.length_scales[i]
                self.kernel_params = np.array([l1])
                nlogml[i] = eval_log_marginal()

            # find the minimum
            min_idx = np.argwhere(nlogml == np.amin(nlogml))[0]
            print(f"Updated length scale to [{self.length_scales[min_idx]}].")
            self.kernel_params = np.array([self.length_scales[min_idx]])

            plot_ARD_LL_1d(nlogml, self.kernel_params, self.length_scales, self.name, self.iteration)
        

    def kernel_rbf(self, x1, x2):
        """
        Return the similarity measure between two points.
        """

        # broadcast to efficiently evaluate points --> this means we don't need
        # nested loops!
        if x1.shape == x2.shape:
            # add jitter to result
            jitter = True
        else:
            jitter = False

        # we have 2 (or more) features
        x1 = x1[:, None, :]
        x2 = x2[None, :, :]

        # set appropriate kernel parameters
        l_scale = self.kernel_params

        # calculate the squared distance between each point
        distance        = (x1 - x2)**2
        scaled_distance = distance / l_scale**2
        rbf             = np.exp(-0.5 * np.sum(scaled_distance, axis = 2)) 
        
        if jitter == True:
            # add jitter noise to diagonals for numerical stability
            return rbf + 1e-4 * np.eye(len(x1))
        else:
            return rbf

    def lower_confidence_bound(self, explore = 4):
        """
        Lower confidence bound acquisition function. Decide the best place to sample
        next.
        """

        print(f"Cov: {self.cov_func.shape}, Mean: {self.mean_func.shape}")
        self.acq_func_eval = explore * self.cov_func - self.mean_func
        print(self.acq_func_eval)
        # return the idx of the maximum value of this functions
        return np.argwhere(self.acq_func_eval == np.amax(self.acq_func_eval))[0]