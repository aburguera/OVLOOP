#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : transform2d
# Description : 2D rigid transformations [x,y,o]: composition, inversion,
#               point transformation and least squares fitting. Poses can be
#               provided as 1D arrays of length 3 or as 3xN arrays (one pose
#               per column).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np

###############################################################################
# FUNCTIONS
###############################################################################

# =============================================================================
# NORMALIZE_ANGLES
#
# Description : Wraps angles to the interval (-pi,pi].
# Input       : theAngles - Scalar or array of angles (rad).
# Output      : The wrapped angles.
# =============================================================================
def normalize_angles(theAngles):
    return theAngles+(2*np.pi)*np.floor((np.pi-theAngles)/(2*np.pi))

# =============================================================================
# COMPOSE_REFERENCES
#
# Description : Composes two transformations (X1 (+) X2).
# Input       : X1 - Transformation from A to B. Shape (3,) or (3,N).
#               X2 - Transformation from B to C. Shape (3,) or (3,N).
# Output      : X3 - Transformation from A to C. Same shape as the inputs.
# =============================================================================
def compose_references(X1,X2):
    X1=np.asarray(X1,dtype=float)
    X2=np.asarray(X2,dtype=float)
    c=np.cos(X1[2])
    s=np.sin(X1[2])
    X3=np.array([X1[0]+X2[0]*c-X2[1]*s,X1[1]+X2[0]*s+X2[1]*c,normalize_angles(X1[2]+X2[2])])
    return X3

# =============================================================================
# INVERT_REFERENCE
#
# Description : Inverts a transformation ((-)X).
# Input       : X - Transformation from A to B. Shape (3,) or (3,N).
# Output      : Xinv - Transformation from B to A.
# =============================================================================
def invert_reference(X):
    X=np.asarray(X,dtype=float)
    c=np.cos(X[2])
    s=np.sin(X[2])
    return np.array([-X[0]*c-X[1]*s,X[0]*s-X[1]*c,-X[2]])

# =============================================================================
# RELATIVE_MOTION
#
# Description : Computes the motion from pose Xa to pose Xb, that is,
#               (-)Xa (+) Xb.
# Input       : Xa,Xb - Absolute poses. Shape (3,) or (3,N).
# Output      : The relative motion.
# =============================================================================
def relative_motion(Xa,Xb):
    return compose_references(invert_reference(Xa),Xb)

# =============================================================================
# COMPOSE_POINT
#
# Description : Applies a transformation to a set of 2D points.
# Input       : X - Transformation [x,y,o].
#               thePoints - 2xN array of points.
# Output      : The transformed 2xN points.
# =============================================================================
def compose_point(X,thePoints):
    thePoints=np.asarray(thePoints,dtype=float).reshape(2,-1)
    c=np.cos(X[2])
    s=np.sin(X[2])
    return np.array([X[0]+c*thePoints[0]-s*thePoints[1],X[1]+s*thePoints[0]+c*thePoints[1]])

# =============================================================================
# LEAST_SQUARES_RIGID
#
# Description : Closed form least squares rigid transformation (rotation and
#               translation, no scale) that maps curPoints onto refPoints,
#               that is, refPoints ~= compose_point(X,curPoints).
# Input       : refPoints - 2xN reference points.
#               curPoints - 2xN points to align.
# Output      : X - Transformation [x,y,o].
# =============================================================================
def least_squares_rigid(refPoints,curPoints):
    refMean=np.mean(refPoints,axis=1,keepdims=True)
    curMean=np.mean(curPoints,axis=1,keepdims=True)
    refCentered=refPoints-refMean
    curCentered=curPoints-curMean
    sXX=np.sum(curCentered[0]*refCentered[0]+curCentered[1]*refCentered[1])
    sXY=np.sum(curCentered[0]*refCentered[1]-curCentered[1]*refCentered[0])
    theAngle=np.arctan2(sXY,sXX)
    c=np.cos(theAngle)
    s=np.sin(theAngle)
    tX=refMean[0,0]-(c*curMean[0,0]-s*curMean[1,0])
    tY=refMean[1,0]-(s*curMean[0,0]+c*curMean[1,0])
    return np.array([tX,tY,theAngle])

# =============================================================================
# ALIGN_TRAJECTORIES
#
# Description : Finds the rigid transformation that best aligns (least squa-
#               res on positions) an estimated trajectory to the ground truth
#               and applies it to the estimate.
# Input       : Xgt - 3xN ground truth poses.
#               Xest - 3xN estimated poses.
# Output      : XestAligned - 3xN aligned estimated poses.
#               Xalign - The aligning transformation.
# =============================================================================
def align_trajectories(Xgt,Xest):
    Xalign=least_squares_rigid(Xgt[:2],Xest[:2])
    XestAligned=compose_references(Xalign.reshape(3,1),Xest)
    return XestAligned,Xalign

# =============================================================================
# COMPOSE_WITH_COVARIANCE
#
# Description : Composition X1 (+) X2 with first order propagation of the
#               covariances of both (independent) transformations.
# Input       : X1,P1 - Transformation (3,) and its 3x3 covariance.
#               X2,P2 - Transformation (3,) and its 3x3 covariance.
# Output      : X3,P3 - Composed transformation and its covariance.
# =============================================================================
def compose_with_covariance(X1,P1,X2,P2):
    c=np.cos(X1[2])
    s=np.sin(X1[2])
    J1=np.array([[1,0,-X2[0]*s-X2[1]*c],[0,1,X2[0]*c-X2[1]*s],[0,0,1]])
    J2=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return compose_references(X1,X2),J1@P1@J1.T+J2@P2@J2.T

# =============================================================================
# INVERT_WITH_COVARIANCE
#
# Description : Inversion (-)X with first order covariance propagation.
# Input       : X,P - Transformation (3,) and its 3x3 covariance.
# Output      : Xinv,Pinv - Inverse and its covariance.
# =============================================================================
def invert_with_covariance(X,P):
    c=np.cos(X[2])
    s=np.sin(X[2])
    J=np.array([[-c,-s,X[0]*s-X[1]*c],[s,-c,X[0]*c+X[1]*s],[0,0,-1]])
    return invert_reference(X),J@P@J.T
