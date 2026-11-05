#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : SLAMParameters
# Description : Parameters of the SLAM system. The default values are those
#               used in the paper. The two validation thresholds were
#               calibrated on validation data: inlierThreshold is the
#               smallest number of inliers that produced no wrong loop and
#               mapThreshold is the 99% quantile of the squared Mahalanobis
#               distance of the correct loops. They depend on the camera,
#               the sea floor and the odometry, so they should be calibrated
#               again for very different conditions. The distances are ex-
#               pressed in world units (those of the altitude and the odo-
#               metry). The default values correspond to the pixels of the
#               photomosaics used in the paper, so loopSigmaXY must be re-
#               scaled for other units (for example, metres).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
from dataclasses import dataclass

###############################################################################
# CLASS
###############################################################################
@dataclass(frozen=True)
class SLAMParameters:
    excludeWindow:int=10            # Most recent images not searched for loops.
    numCandidates:int=5             # Loop candidates per image (top-N).
    optimizeEvery:int=10            # Minimum images between optimizations.
    loopSigmaXY:float=3.0           # Loop translation std (world units).
    loopSigmaTheta:float=0.01       # Loop rotation std (rad).
    inlierThreshold:int=7           # Minimum RANSAC inliers (INL test).
    mapThreshold:float=54.67        # Calibrated map consistency threshold.
