#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : ImageRegistrator
# Description : Geometric verification of loop candidates. Contrast Limited
#               Adaptive Histogram Equalization (CLAHE) is applied, SIFT
#               features are extracted and matched by brute force with the
#               ratio test and a metric rigid model is robustly estimated
#               with RANSAC. The metric rigid model exploits the bottom-
#               looking camera and the altimeter: keypoints are converted
#               to metric camera coordinates using the altitude, so that the
#               motion between the two images is a 2D rigid transformation
#               (3 degrees of freedom) estimated from 2-point samples.
#               Image coordinate convention: the camera coordinates of pixel
#               (u,v) are ((u-cu)*s,(v-cv)*s), where (cu,cv) is the image
#               centre and s the world units per pixel.
#               The camera is assumed to be an ideal pinhole camera pointing
#               straight down over a flat sea floor, so the lens distortion
#               of real images must be removed beforehand. Pitch, roll and
#               the relief of the sea floor are not modelled.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import cv2
import numpy as np
from transform2d import least_squares_rigid,compose_point

###############################################################################
# CLASS
###############################################################################
class ImageRegistrator:

    # CLASS CONSTANTS
    MIN_MATCHES=6               # Below this, registration is not attempted.
    MIN_INLIERS=3               # Below this, registration fails.

    # =========================================================================
    # CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT CREATE THE DETECTORS.
    # Input  : hOpening - Camera horizontal opening (rad).
    #          maxFeatures - Maximum number of SIFT features.
    #          ratioThreshold - Ratio test threshold.
    #          ransacThreshold - RANSAC inlier threshold (pixels).
    #          ransacIterations - Number of RANSAC hypotheses.
    #          randomSeed - Seed of the RANSAC sampling.
    # =========================================================================
    def __init__(self,hOpening=np.pi/2,maxFeatures=1000,ratioThreshold=0.8,ransacThreshold=3.0,ransacIterations=2000,randomSeed=0):
        self.hOpening=hOpening
        self.maxFeatures=maxFeatures
        self.ratioThreshold=ratioThreshold
        self.ransacThreshold=ransacThreshold
        self.ransacIterations=ransacIterations
        self.theRandom=np.random.default_rng(randomSeed)
        self._theDetector=None
        self._theMatcher=None
        self._theCLAHE=None

    # =========================================================================
    # CREATE
    #
    # Description : Creates the OpenCV detector, matcher and CLAHE.
    # =========================================================================
    def create(self):
        self._theDetector=cv2.SIFT_create(nfeatures=self.maxFeatures)
        self._theMatcher=cv2.BFMatcher(cv2.NORM_L2)
        self._theCLAHE=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

    # =========================================================================
    # EXTRACT
    #
    # Description : Extracts the local features of an image.
    # Input       : theImage - HxWx3 uint8 RGB image.
    # Output      : theFeatures - Dict with 'points' (Nx2 float32 pixel
    #                   coordinates), 'descriptors' (Nx128) and 'size' (H,W).
    # =========================================================================
    def extract(self,theImage):
        if self._theDetector is None:
            self.create()
        grayImage=self._theCLAHE.apply(cv2.cvtColor(theImage,cv2.COLOR_RGB2GRAY))
        theKeypoints,theDescriptors=self._theDetector.detectAndCompute(grayImage,None)
        thePoints=np.float32([k.pt for k in theKeypoints]).reshape(-1,2)
        if theDescriptors is None:
            theDescriptors=np.zeros((0,128),dtype=np.float32)
        return {'points':thePoints,'descriptors':theDescriptors.astype(np.float32),'size':grayImage.shape}

    # =========================================================================
    # REGISTER
    #
    # Description : Registers a current image against a reference image and
    #               estimates the camera motion from the reference to the
    #               current one.
    # Input       : refFeatures,curFeatures - As returned by extract.
    #               refAltitude,curAltitude - Altimeter readings (world
    #                   units). Used to convert pixels to world units.
    # Output      : theResult - Dict with:
    #                 isValid    : A model could be estimated.
    #                 numMatches : Matches after the ratio test.
    #                 numInliers : RANSAC inliers.
    #                 theMotion  : [x,y,o] camera motion (world units).
    # =========================================================================
    def register(self,refFeatures,curFeatures,refAltitude,curAltitude):
        theResult={'isValid':False,'numMatches':0,'numInliers':0,'theMotion':np.zeros(3)}
        if self._theMatcher is None:
            self.create()
        refPoints,curPoints=self._match(refFeatures,curFeatures)
        theResult['numMatches']=refPoints.shape[0]
        if refPoints.shape[0]<self.MIN_MATCHES:
            return theResult

        # Pixels to metric camera coordinates
        numRows,numCols=refFeatures['size']
        theCenter=np.array([(numCols-1)/2,(numRows-1)/2])
        refScale=self.get_scale(refAltitude,numCols)
        curScale=self.get_scale(curAltitude,numCols)
        refMetric=((refPoints-theCenter)*refScale).T
        curMetric=((curPoints-theCenter)*curScale).T

        # Robust estimation of the rigid motion
        theMotion,isInlier=self._ransac_rigid(refMetric,curMetric,self.ransacThreshold*refScale)
        if theMotion is None or np.sum(isInlier)<self.MIN_INLIERS:
            return theResult
        theResult['isValid']=True
        theResult['numInliers']=int(np.sum(isInlier))
        theResult['theMotion']=theMotion
        return theResult

    # =========================================================================
    # GET_SCALE
    #
    # Description : World units per pixel of a bottom-looking camera.
    # Input       : theAltitude - Altitude (world units).
    #               numColumns - Image width (pixels).
    # Output      : The scale factor.
    # =========================================================================
    def get_scale(self,theAltitude,numColumns):
        return float(2*theAltitude*np.tan(self.hOpening/2)/numColumns)

    # =========================================================================
    # _MATCH
    #
    # Description : Descriptor matching with the ratio test.
    # Input       : refFeatures,curFeatures - Feature dicts.
    # Output      : refPoints,curPoints - Mx2 matched pixel coordinates.
    # =========================================================================
    def _match(self,refFeatures,curFeatures):
        emptyPoints=np.zeros((0,2),dtype=np.float32)
        if refFeatures['descriptors'].shape[0]<2 or curFeatures['descriptors'].shape[0]<2:
            return emptyPoints,emptyPoints
        theMatches=self._theMatcher.knnMatch(curFeatures['descriptors'],refFeatures['descriptors'],k=2)
        goodMatches=[m[0] for m in theMatches if len(m)==2 and m[0].distance<self.ratioThreshold*m[1].distance]
        if len(goodMatches)==0:
            return emptyPoints,emptyPoints
        curPoints=curFeatures['points'][[m.queryIdx for m in goodMatches]]
        refPoints=refFeatures['points'][[m.trainIdx for m in goodMatches]]
        return refPoints,curPoints

    # =========================================================================
    # _RANSAC_RIGID
    #
    # Description : 2-point RANSAC for 2D rigid motions in metric units.
    #               All the hypotheses are evaluated at once: each one is the
    #               rigid motion defined by two random matches. The largest
    #               consensus set is refined by least squares.
    # Input       : refMetric,curMetric - 2xM matched metric points.
    #               theThreshold - Inlier threshold (world units).
    # Output      : theMotion - [x,y,o] (None on failure).
    #               isInlier - Boolean inlier mask.
    # =========================================================================
    def _ransac_rigid(self,refMetric,curMetric,theThreshold):
        numPoints=refMetric.shape[1]
        firstIndex=self.theRandom.integers(0,numPoints,self.ransacIterations)
        secondIndex=self.theRandom.integers(0,numPoints,self.ransacIterations)
        isDistinct=firstIndex!=secondIndex
        firstIndex=firstIndex[isDistinct]
        secondIndex=secondIndex[isDistinct]

        # Closed form rotation and translation of each hypothesis
        refDelta=refMetric[:,secondIndex]-refMetric[:,firstIndex]
        curDelta=curMetric[:,secondIndex]-curMetric[:,firstIndex]
        theAngles=np.arctan2(refDelta[1],refDelta[0])-np.arctan2(curDelta[1],curDelta[0])
        c=np.cos(theAngles)
        s=np.sin(theAngles)
        tX=refMetric[0,firstIndex]-(c*curMetric[0,firstIndex]-s*curMetric[1,firstIndex])
        tY=refMetric[1,firstIndex]-(s*curMetric[0,firstIndex]+c*curMetric[1,firstIndex])

        # Residuals of all the matches for all the hypotheses
        predX=tX[:,None]+c[:,None]*curMetric[0][None,:]-s[:,None]*curMetric[1][None,:]
        predY=tY[:,None]+s[:,None]*curMetric[0][None,:]+c[:,None]*curMetric[1][None,:]
        theResiduals=np.hypot(predX-refMetric[0][None,:],predY-refMetric[1][None,:])
        numInliers=np.sum(theResiduals<theThreshold,axis=1)
        if numInliers.size==0 or numInliers.max()<self.MIN_INLIERS:
            return None,None
        isInlier=theResiduals[np.argmax(numInliers)]<theThreshold

        # Refinement: least squares on the consensus set, twice
        for _ in range(2):
            theMotion=least_squares_rigid(refMetric[:,isInlier],curMetric[:,isInlier])
            theResiduals=np.linalg.norm(compose_point(theMotion,curMetric)-refMetric,axis=0)
            newInlier=theResiduals<theThreshold
            if newInlier.sum()<self.MIN_INLIERS:
                break
            isInlier=newInlier
        theMotion=least_squares_rigid(refMetric[:,isInlier],curMetric[:,isInlier])
        return theMotion,isInlier
