''' sensor_models.py
Sensor Models
This code attemps to replicate what a sensor will do (and is a very basic
implementation). Our sensor is a simple radar or lidar sensor that gives us 
range and azimuth measurements

AUTHOR: ABHII SUNDARAM, CREATED 06/03/2019
'''
import numpy as np
from util import *
from track_util import draw_from_cov
Pd = 1.0

''' generate_meas
This function generates measurements for each track using truth 
Inputs:
truth: a list of dictionaries (LoD) with target position
R    : measurement covariance matrix
Outputs:
meas : a list of dictionaries (LoD) with measurements for each target
'''
def generate_meas(truth, R, k, perfect_assoc):
    meas = []
    for x_truth in truth:
        z = dict()
        x_truth_k = subfield(x_truth, k)

        if np.random.randn() <= Pd:

            # Generate the measurement space truth
            rng_true = np.sqrt(x_truth_k['xpos_enu']**2 + x_truth_k['ypos_enu']**2)
            az_true = np.arctan2(x_truth_k['xpos_enu'], x_truth_k['ypos_enu'])

            # Draw from measurement covariance to generate measurement noise
            rng_noise, az_noise = draw_from_cov(R)

            # Construct the measurement 
            rng = rng_true + rng_noise
            az = az_true + az_noise

            z['range'] = rng
            z['azimuth'] = az

            # If we are assuming perfect measurement assocation, give this 
            # measurement an id as well, 
            if perfect_assoc == True:
                z['id'] = x_truth['id']

            # If this is the first time we are receiving measurements, 
            # then allow each measurement to initiate a new track (0 = init new track, -1 = dont use this meas)
            if k == 0:
                z['id'] = 0

            meas.append(z)
    return meas
