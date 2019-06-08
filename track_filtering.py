''' track_filtering.py
Kalman Filter Tracks
This code contains measurement filtering update logic for the tracker

AUTHOR: ABHII SUNDARAM, CREATED 06/03/2019
'''
import numpy as np
from util import *

def update_tracks(pred_tracks, meas, R, dim):
    tracks = []
    for x_pred_track in pred_tracks:
        meas_x = findId(meas, x_pred_track['id'])
        x_pred, P_pred = track_dict2nparray(x_pred_track)
        H = gen_H(x_pred)
        x_est, P_est  = ekf_update(x_pred, P_pred, meas_x, H, R, dim)
        x_track = track_nparray2dict(x_est, P_est, x_pred_track['id'])

        tracks.append(x_track)

    # Measurements with id = 0 are to be initialized as new tracks
    meas_new_init = findId(meas, 0)
    for z in meas_new_init:
        x_pred = np.array([1., 1., 0., 0.], dtype = float)
        x_pred = x_pred[:, np.newaxis]
        P_pred = np.eye((dim), dtype=float)
        H = gen_H(x_pred)
        x_est, P_est  = ekf_update(x_pred, P_pred, [z], H, R, dim)
        x_track = track_nparray2dict(x_est, P_est, len(tracks)+1)
        z['id'] = len(tracks)+1

        tracks.append(x_track)
        meas.append(z)
    upd_meas = findId(meas, 0, exclude = True)
    return tracks, upd_meas


def ekf_update(x_pred, P_pred, meas, H, R, dim):

    x_est = x_pred
    P_est = P_pred
    
    # Perform an EKF step for each measurement that needs to update this track (there should really only be 1 per dwell)
    for k in range(0, len(meas)):
        z = meas_dict2nparray(meas[k])
        v = z - h_obs(x_pred)
        S = np.dot(np.dot(H, P_pred) , H.T) + R
        K = np.dot(np.dot(P_pred, H.T) , np.linalg.inv(S))
        x_est = x_pred + np.dot(K, v)

        I = np.eye((dim), dtype = float)
        P_est = np.dot(I - np.dot(K, H), P_pred)
        # P_est = np.dot(np.dot(I - np.dot(K, H), P_pred) , (I - np.dot(K, H)).T) + np.dot(np.dot(K, R) , K.T)

        x_pred = x_est
        P_pred = P_est

    return x_est, P_est

def h_obs(x_pred):
    rng = np.sqrt(x_pred[0]**2 + x_pred[1]**2)
    az = np.arctan2(x_pred[0], x_pred[1])
    z = np.array([rng, az])
    return z

def gen_H(x_pred):
    x = x_pred[0]
    y = x_pred[1]
    r = np.sqrt(x**2 + y**2)
    # For a radar measurement of (range, azimuth),
    drdX = np.array([x/r, y/r, 0, 0])
    dazdX = np.array([y/r**2, -x/r**2, 0, 0])
    H = np.array([drdX, dazdX], dtype = float)
    return H
