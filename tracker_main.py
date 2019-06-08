''' tracker_main.py
Tracker Main 
This code attemps to create a simple multiple target tracker (MTT) with 
different configurable filter update and data association modules. 
This is the main body that runs the tracker.

AUTHOR: ABHII SUNDARAM, CREATED 06/03/2019
'''
import numpy as np
from sensor_models import *
from track_propagation import propagate_tracks
from track_filtering import update_tracks
from track_util import compute_metrics, generate_plots

# Global Variables
n = 100
ntarg = 1
dt = .1
R = np.eye((2), dtype = float)
R[1,1] = 0.0003046
dim = 4

''' main
This funtion runs the tracker
Inputs: 
n : Number of track updates
Outputs:
(None)
'''
def main():

    # Set up the simulation itself
    truth = setup_simulation(n, ntarg, dt)
    tracks = []
    tracks_hist = []

    # Run the tracker fpr n updates
    for k in range(0, n):

        # Receive the measurements and assum1e perfect data assocation
        meas = generate_meas(truth, R, k, perfect_assoc = True)

        # Data association (SKIP for now)

        # Propagate tracks to measurement time
        pred_tracks = propagate_tracks(tracks, k, dt, model = 'cv', q = 0.3)

        # Update tracks
        tracks, meas = update_tracks(pred_tracks, meas, R, dim)

        # Take our tracks and put them into our track history
        tracks_hist = addtrack2Hist(tracks_hist, tracks)

        # Take our measurements and add them to each track that they correlated to
        # and add this to our track history LoD
        tracks_hist = addmeas2trackHist(tracks_hist, meas)

    # Compute track performance metrics
    met = compute_metrics(tracks_hist, truth)
    generate_plots(tracks_hist, truth, meas, met, n, dt)
    return

''' setup_simulation
Set up the simulation, creating truth for simulated targets
Input:
num_updates : Number of truth samples to generate for each target
num_targets : Number of targets to generate truth for
dt          : Fixed time interval between updates
Output: 
truth : list of dictionaries (LoD) of target position truth information
        'Xposvel_enu' is a numpy array of (x, y, vx, vy) position & velocity
        information in the ENU coordinate frame
'''
def setup_simulation(num_updates, num_targets, dt):
    truth = []
    for i in range(0, num_targets):
        x_truth = dict()

        x_truth['id'] = i+1

        x_truth['xpos_enu'] = np.zeros((num_updates), dtype=float)
        x_truth['ypos_enu'] = np.arange(0, 10., 1.*dt, dtype=float)
        x_truth['xvel_enu'] = np.zeros((num_updates), dtype = float)
        x_truth['yvel_enu'] = 1*np.ones((num_updates), dtype = float)

        truth.append(x_truth)

    return truth
    

if __name__ == "__main__":
    main()
