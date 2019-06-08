''' track_util.py
Tracking Utilities
This code has additional tracking related functions that will be useful for 
both tracking systems and performance analysis

AUTHOR: ABHII SUNDARAM, CREATED 06/04/2019
'''

import numpy as np
from util import *
import matplotlib.pyplot as plt

''' draw_from_cov
This function draws a random realization from a given covariance matrix
that represents the covariance of a vector valued Gaussian random variable
Inputs: 
P : Covariance Matrix from which to draw (N X N)
Outputs:
x (not shown) : a vector of random variable of length N 
'''
def draw_from_cov(P):
    U, S, V = np.linalg.svd(P)
    S = np.diag(S)
    return np.dot( np.dot(U, np.sqrt(S)) , np.random.randn(2,1) )

def compute_metrics(tracks_hist, truth):
    met  = dict()
    nees2 = []
    for x_track in tracks_hist:
        x_truth = findId(truth, x_track['id'])[0]
        x_est, P_est = track_dict2nparray(x_track)
        iP_est = np.zeros((P_est.shape))
        for k in range(0, P_est.shape[2]):
            iP_est[:,:,k] = np.linalg.inv(P_est[:,:,k]) 
        x_true = truth_dict2nparray(x_truth)
        x_tilde = x_est - x_true
        nees = np.sum(np.dot(x_tilde.T, iP_est) * x_tilde, axis = (1,2))

        nees2_x = np.zeros((P_est.shape[2]))
        for k in range(0, P_est.shape[2]):
            x_tilde = x_est[:,k] - x_true[:,k]
            iP = np.linalg.inv(P_est[:,:,k]) 
            nees2_x[k] = np.dot(np.dot(x_tilde.T, iP) , x_tilde)
        nees2.append(nees2_x)
    met['nees'] = nees2
    return met

def generate_plots(tracks_hist, truth, meas, met, n, dt):

    for x_track in tracks_hist:
        x_est, P_est = track_dict2nparray(x_track)
        x_enu = x_est[0,:]
        y_enu = x_est[1,:]

        x_truth = findId(truth, x_track['id'])[0]
        x_true = truth_dict2nparray(x_truth)

        meas_x = findId(meas, x_track['id'])[0]
        z = meas_dict2nparray(meas_x)
        rng = z[0,:]
        az = z[0,:]
        x_meas = rng * np.cos(az)
        y_meas = rng * np.sin(az)

        x = x_true[0,:]
        y = x_true[1,:]

        # Plot results
        truthplt = plt.plot(x, y)
        trkplt = plt.plot(x_enu, y_enu)
        plt.xlabel("X ENU (m)")
        plt.ylabel("Y ENU (m)")
        plt.title("2D ENU")
        axis = plt.gca()
        axis.set_xlim((-10, 10))
        axis.set_ylim((-2.5, 12.5))
        measplt = plt.plot(x_meas, y_meas, 'ro')
        plt.setp(measplt, 'markersize', 8)
        plt.setp(trkplt, 'color', 'g', 'linewidth', 10)
        plt.setp(truthplt, 'color', 'k', 'linewidth', 11)
        plt.show()

        t = np.arange(0, n, dtype=float) * dt

        plt.plot(t, x_enu)
        plt.plot(t, x_enu + np.sqrt(P_est[0,0]))
        plt.plot(t, x_enu - np.sqrt(P_est[0,0]))
        plt.xlabel("Time (s)")
        plt.ylabel("X ENU")
        plt.title("X Plot")
        plt.show()

        plt.plot(t, y_enu)
        plt.plot(t, y_enu + np.sqrt(P_est[1,1]))
        plt.plot(t, y_enu - np.sqrt(P_est[1,1]))
        plt.xlabel("Time (s)")
        plt.ylabel("Y ENU")
        plt.title("Y Plot")
        plt.show()

        chibar95 = 13.0 * np.ones((t.shape), dtype=float)

        plt.plot(t, met['nees'][0])
        plt.plot(t, chibar95)
        plt.xlabel("Time (s)")
        plt.ylabel("NEES")
        plt.title("Normalized Estimation Error Squared")
        plt.show()
    return