#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : util
# Description : General purpose utilities.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import sys
import numpy as np
from transform2d import align_trajectories

###############################################################################
# FUNCTIONS
###############################################################################

# =============================================================================
# PROGRESS_BAR
#
# Description : Prints a text progress bar (in place).
# Input       : curValue - Current progress.
#               maxValue - Value corresponding to 100%.
#               barLength - Number of characters of the bar.
#               theLabel - Text printed before the bar.
# =============================================================================
def progress_bar(curValue,maxValue,barLength=40,theLabel=''):
    maxValue=max(maxValue,1)
    theFraction=min(max(curValue/maxValue,0),1)
    numFilled=int(round(barLength*theFraction))
    theBar='#'*numFilled+'-'*(barLength-numFilled)
    sys.stdout.write('\r  %s[%s] %5.1f%%'%(theLabel,theBar,100*theFraction))
    if curValue>=maxValue:
        sys.stdout.write('\n')
    sys.stdout.flush()

# =============================================================================
# COMPUTE_ATE
#
# Description : Absolute Trajectory Error: RMSE of the position errors after
#               aligning the estimate to the ground truth with a rigid
#               transformation.
# Input       : gtPoses - 3xN ground truth poses.
#               estPoses - 3xN estimated poses.
# Output      : The ATE (world units).
# =============================================================================
def compute_ate(gtPoses,estPoses):
    alignedPoses,_=align_trajectories(gtPoses,estPoses)
    return float(np.sqrt(np.mean(np.sum((alignedPoses[:2]-gtPoses[:2])**2,axis=0))))
