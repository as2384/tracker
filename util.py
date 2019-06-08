''' util.py
Utility Functions 
This code attemps has several utility functions that can be used for data parsing

AUTHOR: ABHII SUNDARAM, CREATED 06/03/2019
'''
import numpy as np

''' subfield
This functions gets only the idx values of the dictionary x
and returns it in a new dict
Inputs:
x   : dictionary
idx : indices of each for each key that you want to keep
Outputs:
xnew: dictionary with keys that have only the idx values
'''
def subfield(x, idx):
    xnew = x.copy()
    for key in x.keys():
        if type(x[key]) is np.ndarray:
            xnew[key] = x[key][idx]
        else:
            xnew[key] = x[key]
    return xnew

def truth_dict2nparray(x_truth):
    x = np.array([x_truth['xpos_enu'], x_truth['ypos_enu'], \
                  x_truth['xvel_enu'], x_truth['yvel_enu']])
    return x

def track_dict2nparray(x_track):
    x_est = np.array([x_track['xpos_enu'], x_track['ypos_enu'], \
                      x_track['xvel_enu'], x_track['yvel_enu']])
    
    P_est = np.array([[x_track['xxcov_enu'],  x_track['xycov_enu'],  x_track['xvxcov_enu'],  x_track['xvycov_enu']], \
                      [x_track['xycov_enu'],  x_track['yycov_enu'],  x_track['yvxcov_enu'],  x_track['yvycov_enu']], \
                      [x_track['xvxcov_enu'], x_track['yvxcov_enu'], x_track['vxvxcov_enu'], x_track['vxvycov_enu']], \
                      [x_track['xvycov_enu'], x_track['yvycov_enu'], x_track['vxvycov_enu'], x_track['vyvycov_enu']]])
    return x_est, P_est

def track_nparray2dict(x, P, id):
    x_track = dict()

    x_track['id'] = id

    # State information
    x_track['xpos_enu'] = x[0]
    x_track['ypos_enu'] = x[1]
    x_track['xvel_enu'] = x[2]
    x_track['yvel_enu'] = x[3]

    if P is None:
        return x_track

    # Covariance Information
    x_track['xxcov_enu'] = P[0,0]
    x_track['xycov_enu'] = P[0,1]
    x_track['xvxcov_enu'] = P[0,2]
    x_track['xvycov_enu'] = P[0,3]
    x_track['yycov_enu']  = P[1,1]
    x_track['yvxcov_enu'] = P[1,2]
    x_track['yvycov_enu'] = P[1,3]
    x_track['vxvxcov_enu'] = P[2,2]
    x_track['vxvycov_enu'] = P[2,3]
    x_track['vyvycov_enu'] = P[3,3]

    return x_track

def meas_dict2nparray(meas):
    z = np.array([meas['range'], meas['azimuth']])
    
    return z

def meas_nparray2dict(z, id):
    meas = dict()

    meas['id'] = id
    meas['range'] = z[0]
    meas['azimuth'] = z[1]
    
    return meas

def findId(lod, id, exclude = False):
    new_lod = []
    for x in lod:
        if not exclude and x['id'] == id:
            new_lod.append(x)
        elif exclude and x['id'] != id:
            new_lod.append(x)
    return new_lod

def getAllIds(lod):
    id_list = []
    for x in lod:
        id_list.append(x['id'])
    return np.unique(id_list)

def addtrack2Hist(tracks_hist, tracks):
    new_tracks_hist = []
    tracks_hist_ids = getAllIds(tracks_hist)
    tracks_ids = getAllIds(tracks)
    ids = np.concatenate((tracks_hist_ids, tracks_ids), axis=0)
    idsu = np.unique(ids)

    for idu in idsu:
        tracks_hist_x = findId(tracks_hist, idu)
        tracks_x = findId(tracks, idu)

        if len(tracks_hist_x) == 0:
            new_tracks_hist.append(tracks_x[0])
        elif len(tracks_x) == 0:
            pass
        else:
            new_tracks_hist_x = dict()
            new_tracks_hist_x['id'] = idu
            new_tracks_hist_x['xpos_enu'] = np.append(tracks_hist_x[0]['xpos_enu'], tracks_x[0]['xpos_enu'])
            new_tracks_hist_x['ypos_enu'] = np.append(tracks_hist_x[0]['ypos_enu'], tracks_x[0]['ypos_enu'])
            new_tracks_hist_x['xvel_enu'] = np.append(tracks_hist_x[0]['xvel_enu'], tracks_x[0]['xvel_enu'])
            new_tracks_hist_x['yvel_enu'] = np.append(tracks_hist_x[0]['yvel_enu'], tracks_x[0]['yvel_enu'])

            new_tracks_hist_x['xxcov_enu'] = np.append(tracks_hist_x[0]['xxcov_enu'], tracks_x[0]['xxcov_enu'])
            new_tracks_hist_x['xycov_enu'] = np.append(tracks_hist_x[0]['xycov_enu'], tracks_x[0]['xycov_enu'])
            new_tracks_hist_x['xvxcov_enu'] = np.append(tracks_hist_x[0]['xvxcov_enu'], tracks_x[0]['xvxcov_enu'])
            new_tracks_hist_x['xvycov_enu'] = np.append(tracks_hist_x[0]['xvycov_enu'], tracks_x[0]['xvycov_enu'])
            new_tracks_hist_x['yycov_enu'] = np.append(tracks_hist_x[0]['yycov_enu'], tracks_x[0]['yycov_enu'])
            new_tracks_hist_x['yvxcov_enu'] = np.append(tracks_hist_x[0]['yvxcov_enu'], tracks_x[0]['yvxcov_enu'])
            new_tracks_hist_x['yvycov_enu'] = np.append(tracks_hist_x[0]['yvycov_enu'], tracks_x[0]['yvycov_enu'])
            new_tracks_hist_x['vxvxcov_enu'] = np.append(tracks_hist_x[0]['vxvxcov_enu'], tracks_x[0]['vxvxcov_enu'])
            new_tracks_hist_x['vxvycov_enu'] = np.append(tracks_hist_x[0]['vxvycov_enu'], tracks_x[0]['vxvycov_enu'])
            new_tracks_hist_x['vyvycov_enu'] = np.append(tracks_hist_x[0]['vyvycov_enu'], tracks_x[0]['vyvycov_enu'])

            new_tracks_hist.append(new_tracks_hist_x)

    return new_tracks_hist

def addmeas2trackHist(tracks_hist, meas):
    new_tracks_hist = []
    tracks_hist_ids = getAllIds(tracks_hist)

    for id in tracks_hist_ids:
        tracks_hist_x = findId(tracks_hist, id)[0]
        meas_x = findId(meas, id)[0]

        new_tracks_hist_x = tracks_hist_x.copy()

        new_tracks_hist_x['range'] = np.append(tracks_hist_x[0]['range'], meas_x['range'])
        new_tracks_hist_x['azimuth'] = np.append(tracks_hist_x[0]['azimuth'], meas_x['azimuth']) 

        new_tracks_hist.append(new_tracks_hist_x)

    return new_tracks_hist
