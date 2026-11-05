#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : plotutil
# Description : Plotting utilities.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
from transform2d import align_trajectories

###############################################################################
# FUNCTIONS
###############################################################################

# =============================================================================
# PLOT_SLAM
#
# Description : Plots an estimated trajectory, its loops and, optionally,
#               the ground truth. If the ground truth is provided, the
#               estimate is rigidly aligned to it before plotting.
# Input       : theAxis - Matplotlib axis.
#               estPoses - 3xN estimated poses.
#               theLoops - List of (candidate,query) image indices.
#               gtPoses - 3xN ground truth poses or None.
# =============================================================================
def plot_slam(theAxis,estPoses,theLoops,gtPoses=None):
    if gtPoses is not None:
        estPoses,_=align_trajectories(gtPoses[:,:estPoses.shape[1]],estPoses)
        theAxis.plot(gtPoses[0],gtPoses[1],color='0.75',linewidth=2,label='Ground truth')
    for c,t in theLoops:
        theAxis.plot([estPoses[0,c],estPoses[0,t]],[estPoses[1,c],estPoses[1,t]],color='b',linewidth=0.5)
    theAxis.plot(estPoses[0],estPoses[1],color='k',linewidth=1,label='Estimate')
    theAxis.set_aspect('equal')
    theAxis.invert_yaxis()
    theAxis.legend(loc='best')

# =============================================================================
# PLOT_POSES
#
# Description : Plots a trajectory with arrows showing some orientations.
# Input       : theAxis - Matplotlib axis.
#               thePoses - 3xN poses.
#               arrowEvery - Draw one arrow every arrowEvery poses.
#               arrowLength - Length of the arrows (world units).
# =============================================================================
def plot_poses(theAxis,thePoses,arrowEvery=25,arrowLength=40.0):
    theAxis.plot(thePoses[0],thePoses[1],color='k',linewidth=1)
    theIndices=np.arange(0,thePoses.shape[1],arrowEvery)
    theAxis.quiver(thePoses[0,theIndices],thePoses[1,theIndices],arrowLength*np.cos(thePoses[2,theIndices]),arrowLength*np.sin(thePoses[2,theIndices]),angles='xy',scale_units='xy',scale=1,color='r',width=0.004)
    theAxis.set_aspect('equal')
    theAxis.invert_yaxis()
