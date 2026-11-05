#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : DataSimulator
# Description : Convenience class that simulates an AUV endowed with a
#               bottom-looking camera, an altimeter and an odometer from a
#               UCAMGEN dataset (https://github.com/aburguera/UCAMGEN). The
#               images are those of the dataset. The odometry is obtained by
#               perturbing the ground truth motions with Gaussian noise
#               proportional to the travelled distance and with a heading
#               bias, which produces the curved drift typical of dead reck-
#               oning. The altimeter has relative Gaussian noise.
#               The default noise is that of the main experiments of the
#               paper.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import os
import numpy as np
from PIL import Image
from transform2d import relative_motion,compose_references

class DataSimulatorError(Exception):
    pass

###############################################################################
# CLASS
###############################################################################
class DataSimulator:

    # CLASS CONSTANTS
    IMAGE_FOLDER='IMAGES'               # Images sub-folder.
    IMAGE_PATTERN='IMAGE%05d.png'       # 1-based image file names.
    BORDER_REPAIR=2                     # Border pixels to repair.
    MIN_SIGMA=1e-3                      # Minimum odometric std.

    # =========================================================================
    # CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT LOAD THE DATASET.
    # Input  : datasetPath - UCAMGEN dataset folder.
    #          imageFolder - Alternative images folder (for example, degraded
    #                        images) or None to use the dataset images.
    #          relativeSigma - Odometric translation std (fraction of the
    #                        travelled distance).
    #          absoluteSigma - Odometric translation std (world units/step).
    #          angularSigma - Odometric rotation std (rad/step).
    #          angularBias - Heading bias (rad/step). Its sign is random.
    #          altimeterSigma - Altimeter std (fraction of the altitude).
    #          randomSeed - Seed of the simulated noise.
    # =========================================================================
    def __init__(self,datasetPath,imageFolder=None,relativeSigma=0.08,absoluteSigma=1.0,angularSigma=0.01,angularBias=0.003,altimeterSigma=0.01,randomSeed=0):
        self.datasetPath=datasetPath
        self.imageFolder=os.path.join(datasetPath,self.IMAGE_FOLDER) if imageFolder is None else imageFolder
        self.relativeSigma=relativeSigma
        self.absoluteSigma=absoluteSigma
        self.angularSigma=angularSigma
        self.angularBias=angularBias
        self.altimeterSigma=altimeterSigma
        self.randomSeed=randomSeed
        self.numImages=0
        self.curIndex=-1

    # =========================================================================
    # LOAD
    #
    # Description : Loads the dataset and simulates the sensors.
    # =========================================================================
    def load(self):
        posesFile=os.path.join(self.datasetPath,'POSES.csv')
        if not os.path.exists(posesFile):
            raise DataSimulatorError('%s is not a UCAMGEN dataset'%self.datasetPath)
        rawPoses=np.loadtxt(posesFile,delimiter=',',ndmin=2)
        # UCAMGEN uses 1-based (Matlab) pixel coordinates
        self.gtPoses=rawPoses[:3].copy()
        self.gtPoses[:2]-=1
        self.gtAltitudes=rawPoses[3].copy()
        self.numImages=self.gtPoses.shape[1]
        self.hOpening=float(np.loadtxt(os.path.join(self.datasetPath,'PARAMS.csv'),delimiter=',')[0])
        theRandom=np.random.default_rng(self.randomSeed)
        self.theOdometry,self.odoCovariances=self._simulate_odometry(theRandom)
        self.theAltitudes=self.gtAltitudes*(1+theRandom.normal(0,self.altimeterSigma,self.numImages))
        self.curIndex=-1

    # =========================================================================
    # FETCH
    #
    # Description : Advances to the next image.
    # Output      : False if there are no more images, True otherwise.
    # =========================================================================
    def fetch(self):
        if self.curIndex+1>=self.numImages:
            return False
        self.curIndex+=1
        return True

    # =========================================================================
    # GET_IMAGE / GET_ALTITUDE / GET_ODOMETRY
    #
    # Description : Data of the current image.
    # Output      : HxWx3 uint8 RGB image / altitude / odometric motion from
    #               the previous image and its 3x3 covariance (None,None for
    #               the first image).
    # =========================================================================
    def get_image(self):
        theImage=np.array(Image.open(os.path.join(self.imageFolder,self.IMAGE_PATTERN%(self.curIndex+1))).convert('RGB'))
        # UCAMGEN rotates and crops the viewports. Due to rounding, the
        # outermost rows and columns may contain black pixels. They are
        # replaced by the closest inner ones.
        b=self.BORDER_REPAIR
        theImage[:b]=theImage[b]
        theImage[-b:]=theImage[-b-1]
        theImage[:,:b]=theImage[:,b:b+1]
        theImage[:,-b:]=theImage[:,-b-1:-b]
        return theImage

    def get_altitude(self):
        return float(self.theAltitudes[self.curIndex])

    def get_odometry(self):
        if self.curIndex==0:
            return None,None
        return self.theOdometry[:,self.curIndex-1].copy(),self.odoCovariances[self.curIndex-1].copy()

    # =========================================================================
    # GET_GROUND_TRUTH
    #
    # Description : Ground truth poses (for evaluation and plotting only).
    # Output      : 3xN poses.
    # =========================================================================
    def get_ground_truth(self):
        return self.gtPoses.copy()

    # =========================================================================
    # GET_DEAD_RECKONING
    #
    # Description : Poses obtained by integrating the simulated odometry
    #               from the ground truth initial pose.
    # Output      : 3xN poses.
    # =========================================================================
    def get_dead_reckoning(self):
        thePoses=np.zeros((3,self.numImages))
        thePoses[:,0]=self.gtPoses[:,0]
        for k in range(self.numImages-1):
            thePoses[:,k+1]=compose_references(thePoses[:,k],self.theOdometry[:,k])
        return thePoses

    # =========================================================================
    # _SIMULATE_ODOMETRY
    #
    # Description : Noisy relative motions between consecutive poses and
    #               their covariances. The bias is not included in the
    #               covariance model, since it is unknown to the vehicle,
    #               but its magnitude inflates the rotational covariance.
    # Input       : theRandom - numpy Generator.
    # Output      : theOdometry - 3x(N-1) noisy motions.
    #               theCovariances - (N-1)x3x3 covariances.
    # =========================================================================
    def _simulate_odometry(self,theRandom):
        gtMotions=relative_motion(self.gtPoses[:,:-1],self.gtPoses[:,1:])
        numMotions=gtMotions.shape[1]
        linearSigma=self.relativeSigma*np.hypot(gtMotions[0],gtMotions[1])+self.absoluteSigma
        angularSigma=np.full(numMotions,self.angularSigma)
        angularBias=self.angularBias*theRandom.choice([-1.0,1.0])
        theOdometry=gtMotions.copy()
        theOdometry[0]+=theRandom.normal(0,1,numMotions)*linearSigma
        theOdometry[1]+=theRandom.normal(0,1,numMotions)*linearSigma
        theOdometry[2]+=theRandom.normal(0,1,numMotions)*angularSigma+angularBias
        theCovariances=np.zeros((numMotions,3,3))
        theCovariances[:,0,0]=np.maximum(linearSigma,self.MIN_SIGMA)**2
        theCovariances[:,1,1]=np.maximum(linearSigma,self.MIN_SIGMA)**2
        theCovariances[:,2,2]=np.maximum(angularSigma+abs(angularBias),self.MIN_SIGMA*1e-2)**2
        return theOdometry,theCovariances
