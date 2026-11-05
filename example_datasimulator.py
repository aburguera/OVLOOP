#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_datasimulator
# Description : Usage example of the DataSimulator. A UCAMGEN dataset is
#               loaded, some images are shown and the ground truth is
#               plotted together with the dead reckoning obtained from the
#               simulated (noisy) odometry.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import matplotlib.pyplot as plt
from datasimulator import DataSimulator
from plotutil import plot_poses

###############################################################################
# PARAMETERS
###############################################################################
DATASET_PATH='DATASET'          # UCAMGEN dataset folder. Set your own.
NUM_IMAGES=4                    # Number of images to show.

###############################################################################
# MAIN CODE
###############################################################################

# Load the dataset and simulate the sensors
print('[ LOADING ]')
theSimulator=DataSimulator(DATASET_PATH)
theSimulator.load()
print('  * %d IMAGES, HORIZONTAL OPENING %.2f RAD'%(theSimulator.numImages,theSimulator.hOpening))

# Show the first images with their altitude and odometry
print('[ READING ]')
theFigure,theAxes=plt.subplots(1,NUM_IMAGES,figsize=(4*NUM_IMAGES,3))
for curAxis in theAxes:
    theSimulator.fetch()
    theOdometry,_=theSimulator.get_odometry()
    curAxis.imshow(theSimulator.get_image())
    curAxis.set_title('ALTITUDE %.1f'%theSimulator.get_altitude())
    curAxis.axis('off')
    print('  * IMAGE %d : ODOMETRY %s'%(theSimulator.curIndex,'NONE' if theOdometry is None else '[%.1f,%.1f,%.3f]'%tuple(theOdometry)))

# Ground truth versus dead reckoning
theFigure,theAxes=plt.subplots(1,2,figsize=(12,5))
plot_poses(theAxes[0],theSimulator.get_ground_truth())
theAxes[0].set_title('Ground truth')
plot_poses(theAxes[1],theSimulator.get_dead_reckoning())
theAxes[1].set_title('Dead reckoning')
plt.show()
