''' track_propagation.py
Propagate/Predict Tracks
This code contains prediction logic for the tracker

AUTHOR: ABHII SUNDARAM, CREATED 06/03/2019
'''
import numpy as np
from util import *

def propagate_tracks(tracks, k, dt, model, q):
    pred_tracks = []
    for x_track in tracks:
        x_est, P_est = track_dict2nparray(x_track)
        x_pred, P_pred  = predict_track(x_est, P_est, dt, model, q)
        x_pred_track = track_nparray2dict(x_pred, P_pred, x_track['id'])
        pred_tracks.append(x_pred_track)

    return pred_tracks

def predict_track(x_est, P_est, dt, model, q):

    # Generate state transition matrix and process noise covariance matrix
    F, Q = gen_FQ(dt, model, q)

    # Predict state
    x_pred = np.dot(F, x_est)

    # Predict covariance 
    P_pred = np.dot(F, np.dot(P_est, F.T)) + Q

    return x_pred, P_pred


def gen_FQ(dt, model, q):

    if model == 'cv':
        F = np.array([[1, 0, dt, 0], \
            [0, 1, 0, dt], \
            [0, 0, 1, 0], \
            [0, 0, 0, 1]]) 
        
        Q = np.array([[dt**3/3, 0, dt**2/2, 0],
                      [0, dt**3/3, 0, dt**2/2],
                      [dt**2/2, 0, dt, 0],
                      [0, dt**2/2, 0, dt]]) * q

    return F, Q

