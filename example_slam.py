#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_slam
# Description : Usage example of the SLAM system. A UCAMGEN dataset is used
#               to simulate an AUV (images, altimeter and noisy odometry),
#               the SLAM processes the images one by one and, at the end,
#               the trajectory error of the SLAM is compared with that of
#               dead reckoning and the result is plotted.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import matplotlib.pyplot as plt
from datasimulator import DataSimulator
from hybridretriever import HybridRetriever
from imageregistrator import ImageRegistrator
from slamparameters import SLAMParameters
from slamsystem import SLAMSystem
from plotutil import plot_slam
from util import progress_bar,compute_ate

###############################################################################
# PARAMETERS
###############################################################################
DATASET_PATH='DATASET'          # UCAMGEN dataset folder. Set your own.
MODELS_PATH='MODELS'            # Folder with the trained models.

###############################################################################
# MAIN CODE
###############################################################################

# Load the data and the models
print('[ LOADING ]')
theSimulator=DataSimulator(DATASET_PATH)
theSimulator.load()
theRetriever=HybridRetriever()
theRetriever.load(MODELS_PATH)
theRegistrator=ImageRegistrator(hOpening=theSimulator.hOpening)
gtPoses=theSimulator.get_ground_truth()
theSLAM=SLAMSystem(SLAMParameters(),theRetriever,theRegistrator,gtPoses[:,0])
print('  * %d IMAGES'%theSimulator.numImages)

# Process the images one by one, as the AUV grabs them
print('[ RUNNING SLAM ]')
while theSimulator.fetch():
    theOdometry,odoCovariance=theSimulator.get_odometry()
    theSLAM.set_input(theSimulator.get_image(),theSimulator.get_altitude(),theOdometry,odoCovariance)
    theSLAM.process()
    progress_bar(theSimulator.curIndex+1,theSimulator.numImages)
theSLAM.optimize()

# Evaluate and plot
print('[ RESULTS ]')
estPoses=theSLAM.get_poses()
print('  * ACCEPTED LOOPS : %d'%len(theSLAM.get_loops()))
print('  * ATE DEAD RECKONING : %.1f'%compute_ate(gtPoses,theSimulator.get_dead_reckoning()))
print('  * ATE SLAM : %.1f'%compute_ate(gtPoses,estPoses))
theFigure,theAxis=plt.subplots(figsize=(8,6))
plot_slam(theAxis,estPoses,theSLAM.get_loops(),gtPoses)
plt.show()
