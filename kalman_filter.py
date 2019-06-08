''' kalman_filter.py 
Kalman Filter
This code attemps to create a simple track fusion filter
using the vanilla Kalman Filter 

AUTHOR: ABHII SUNDARAM, CREATED 05/27/2019
'''
import numpy as np

''' A Vanilla Kalman Filter Implementation '''
def kf_update(n, F, Q, R, H, z):

    # Initialize
    xkk = np.zeros((6, 1))
    Pkk = np.zeros((6, 6))

    for k in range(0, n):
        # Propagation
        xkp1k = np.dot(F, xkk) # State
        Pkp1k = np.dot(F, np.dot(Pkk, F.T)) + Q # Covariance

        # Measurement update
        I = np.linalg.eye(6)
        zpred = np.dot(H, xkp1k)
        S = np.dot(H, np.dot(Pkp1k, H.T)) + R
        K = np.dot(Pkp1k, np.dot(H.T, np.linalg.inv(S)))
        xkp1kp1 = xkp1k + np.dot(K, z - zpred) # State
        Pkp1kp1 = np.dot((I-np.dot(K, H)), Pkp1k) # Covariance


    return xkp1kp1, Pkp1kp1