'''
raw data columns: accel x, accel y, accel z, pitch speed, yaw speed, roll speed

features after preprocess:
-max,min,mean
-max, min, mean high pass filtered
-max,min,mean low pass filtered
-variance raw, variance high pass filtered, low pass filtered
-skewness raw, skewness high pass filtered, skewness low pass filtered
-kurtosis raw, kurtosis high pass filtered, low pass filtered

gesture labels:
0: beats
1: pointf
2: pointl
3: pointr
4: raise
5: shrug
6: thumbsdown
7: thumbsup
8: wave

'''

'''
TODO: EDIT UNITY SO DOESNT ADD BLANK ROW AT BOTTOM OF CSV
'''

import os
import numpy as np
import scipy
from scipy.signal import butter, lfilter

def add_feature_row(path, filename, label):
    training_sample = []
    new_row = []
    new_row = np.asarray(new_row)

    os.chdir(path)
    with open(filename, 'r') as f:
        print(filename)
        training_sample = [[float(val) for val in line.split(',')[:6]] for line in f]

    training_sample = np.asarray(training_sample)

    b,a = butter(1,0.25, btype='highpass')
    accel_x_highfilt = lfilter(b, a, training_sample[:,0])
    accel_y_highfilt = lfilter(b, a, training_sample[:,1])
    accel_z_highfilt = lfilter(b, a, training_sample[:,2])
    pitch_highfilt = lfilter(b, a, training_sample[:,3])
    yaw_highfilt = lfilter(b, a, training_sample[:,4])
    roll_highfilt = lfilter(b, a, training_sample[:,5])

    b,a = butter(1,0.25, btype='lowpass')
    accel_x_lowfilt = lfilter(b, a, training_sample[:,0])
    accel_y_lowfilt = lfilter(b, a, training_sample[:,1])
    accel_z_lowfilt = lfilter(b, a, training_sample[:,2])
    pitch_lowfilt = lfilter(b, a, training_sample[:,3])
    yaw_lowfilt = lfilter(b, a, training_sample[:,4])
    roll_lowfilt = lfilter(b, a, training_sample[:,5])

    #### max of each feature ####
    #loop through rows of each column
    for row in training_sample.T:
        row = np.asarray(row)
        max = np.amax(row)
        new_row = np.append(new_row, max)

    #### min of each feature ####
    for row in training_sample.T:
        row = np.asarray(row)
        min = np.var(row)
        new_row = np.append(new_row, min)

    #### mean of each feature ####
    for row in training_sample.T:
        row = np.asarray(row)
        mean = np.mean(row)
        new_row = np.append(new_row, mean)

    #### max high pass filter ####
    new_row = np.append(new_row, np.amax(accel_x_highfilt))
    new_row = np.append(new_row, np.amax(accel_y_highfilt))
    new_row = np.append(new_row, np.amax(accel_z_highfilt))
    new_row = np.append(new_row, np.amax(pitch_highfilt))
    new_row = np.append(new_row, np.amax(yaw_highfilt))
    new_row = np.append(new_row, np.amax(roll_highfilt))

    #### min high pass filter ####
    new_row = np.append(new_row, np.amin(accel_x_highfilt))
    new_row = np.append(new_row, np.amin(accel_y_highfilt))
    new_row = np.append(new_row, np.amin(accel_z_highfilt))
    new_row = np.append(new_row, np.amin(pitch_highfilt))
    new_row = np.append(new_row, np.amin(yaw_highfilt))
    new_row = np.append(new_row, np.amin(roll_highfilt))

    #### mean high pass filter ####
    new_row = np.append(new_row, np.mean(accel_x_highfilt))
    new_row = np.append(new_row, np.mean(accel_y_highfilt))
    new_row = np.append(new_row, np.mean(accel_z_highfilt))
    new_row = np.append(new_row, np.mean(pitch_highfilt))
    new_row = np.append(new_row, np.mean(yaw_highfilt))
    new_row = np.append(new_row, np.mean(roll_highfilt))

    #### max low pass filter ####
    new_row = np.append(new_row, np.amax(accel_x_lowfilt))
    new_row = np.append(new_row, np.amax(accel_y_lowfilt))
    new_row = np.append(new_row, np.amax(accel_z_lowfilt))
    new_row = np.append(new_row, np.amax(pitch_lowfilt))
    new_row = np.append(new_row, np.amax(yaw_lowfilt))
    new_row = np.append(new_row, np.amax(roll_lowfilt))

    #### min low pass filter ####
    new_row = np.append(new_row, np.amin(accel_x_lowfilt))
    new_row = np.append(new_row, np.amin(accel_y_lowfilt))
    new_row = np.append(new_row, np.amin(accel_z_lowfilt))
    new_row = np.append(new_row, np.amin(pitch_lowfilt))
    new_row = np.append(new_row, np.amin(yaw_lowfilt))
    new_row = np.append(new_row, np.amin(roll_lowfilt))

    #### mean low pass filter ####
    new_row = np.append(new_row, np.mean(accel_x_lowfilt))
    new_row = np.append(new_row, np.mean(accel_y_lowfilt))
    new_row = np.append(new_row, np.mean(accel_z_lowfilt))
    new_row = np.append(new_row, np.mean(pitch_lowfilt))
    new_row = np.append(new_row, np.mean(yaw_lowfilt))
    new_row = np.append(new_row, np.mean(roll_lowfilt))

    #### variance ####
    for row in training_sample.T:
        new_row = np.append(new_row, np.var(np.asarray(row)))

    #### variance of high pass filtered data ####
    new_row = np.append(new_row, np.var(accel_x_highfilt))
    new_row = np.append(new_row, np.var(accel_y_highfilt))
    new_row = np.append(new_row, np.var(accel_z_highfilt))
    new_row = np.append(new_row, np.var(pitch_highfilt))
    new_row = np.append(new_row, np.var(yaw_highfilt))
    new_row = np.append(new_row, np.var(roll_highfilt))

    #### variance of low pass filtered data ####
    new_row = np.append(new_row, np.var(accel_x_lowfilt))
    new_row = np.append(new_row, np.var(accel_y_lowfilt))
    new_row = np.append(new_row, np.var(accel_z_lowfilt))
    new_row = np.append(new_row, np.var(pitch_lowfilt))
    new_row = np.append(new_row, np.var(yaw_lowfilt))
    new_row = np.append(new_row, np.var(roll_lowfilt))

    #### skewness ####
    for row in training_sample.T:
        new_row=np.append(new_row, scipy.stats.skew(np.asarray(row)))

    #### skewness high pass filtered ####
    new_row = np.append(new_row, scipy.stats.skew(accel_x_highfilt))
    new_row = np.append(new_row, scipy.stats.skew(accel_y_highfilt))
    new_row = np.append(new_row, scipy.stats.skew(accel_z_highfilt))
    new_row = np.append(new_row, scipy.stats.skew(pitch_highfilt))
    new_row = np.append(new_row, scipy.stats.skew(yaw_highfilt))
    new_row = np.append(new_row, scipy.stats.skew(roll_highfilt))

    #### skewness low pass filtered ####
    new_row = np.append(new_row, scipy.stats.skew(accel_x_lowfilt))
    new_row = np.append(new_row, scipy.stats.skew(accel_y_lowfilt))
    new_row = np.append(new_row, scipy.stats.skew(accel_z_lowfilt))
    new_row = np.append(new_row, scipy.stats.skew(pitch_lowfilt))
    new_row = np.append(new_row, scipy.stats.skew(yaw_lowfilt))
    new_row = np.append(new_row, scipy.stats.skew(roll_lowfilt))

    #### kurtosis ####
    for row in training_sample.T:
        new_row=np.append(new_row, scipy.stats.kurtosis(np.asarray(row)))

    #### kurtosis high pass filtered ####
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_x_highfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_y_highfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_z_highfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(pitch_highfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(yaw_highfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(roll_highfilt))

    #### kurtosis low pass filtered ####
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_x_lowfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_y_lowfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(accel_z_lowfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(pitch_lowfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(yaw_lowfilt))
    new_row = np.append(new_row, scipy.stats.kurtosis(roll_lowfilt))

    new_row = np.append(new_row, label)

    print(new_row)
    print(new_row.shape)

    return(new_row)

features = add_feature_row('/home/robotlab/classifiers/csv/beats', 'S1beats1.csv', 0)
features = np.asarray(features)

filenames = []
for filename in os.listdir(('/home/robotlab/classifiers/csv/beats')):
    if filename != 's1beats1.csv':
        filenames.append(filename)

for f in filenames:
    features = np.vstack((features, add_feature_row('/home/robotlab/classifiers/csv/beats', f, 0)))



gestures = ['pointf', 'pointl', 'pointr', 'raise', 'shrug', 'thumbsdown', 'thumbsup', 'wave']

label = 1
for gesture in gestures:
    filenames = []
    for filename in os.listdir(('/home/robotlab/classifiers/csv/%s'%(gesture))):
        filenames.append(filename)
    for f in filenames:
        features = np.vstack((features, add_feature_row('/home/robotlab/classifiers/csv/%s'%(gesture), f, label)))
    label+=1
print(features.shape)

os.chdir('/home/robotlab/classifiers')
np.savetxt('5subjects.csv', features, delimiter=',')
