#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : PoseGraph
# Description : 2D pose graph (SE(2)) with odometric and loop closure con-
#               straints, optimized by a sparse Levenberg-Marquardt written
#               from scratch (Jacobians as in Grisetti et al., 2010). The
#               first node is anchored. The error of an edge (i,j) with
#               measurement Z is e = t2v(Z^-1 (Xi^-1 Xj)).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
from scipy.sparse import coo_matrix,identity
from scipy.sparse.linalg import spsolve
from transform2d import normalize_angles,relative_motion

###############################################################################
# CLASS
###############################################################################
class PoseGraph:

    # CLASS CONSTANTS
    ANCHOR_INFORMATION=1e8      # Information of the prior on the first node.

    # =========================================================================
    # CONSTRUCTOR
    # Input  : maxIterations - Maximum optimization iterations.
    # =========================================================================
    def __init__(self,maxIterations=50):
        self.maxIterations=maxIterations
        self.thePoses=np.zeros((3,0))
        self.theEdges=[]

    # =========================================================================
    # ADD_NODE
    #
    # Description : Adds a node with an initial pose estimate.
    # Input       : thePose - [x,y,o].
    # Output      : Index of the node.
    # =========================================================================
    def add_node(self,thePose):
        self.thePoses=np.hstack((self.thePoses,np.asarray(thePose,dtype=float).reshape(3,1)))
        return self.thePoses.shape[1]-1

    # =========================================================================
    # ADD_EDGE
    #
    # Description : Adds a relative pose constraint from node i to node j.
    # Input       : i,j - Node indices.
    #               theMeasurement - Motion [x,y,o] from i to j.
    #               theInformation - 3x3 information matrix.
    # =========================================================================
    def add_edge(self,i,j,theMeasurement,theInformation):
        self.theEdges.append((int(i),int(j),np.asarray(theMeasurement,dtype=float).copy(),np.asarray(theInformation,dtype=float).copy()))

    # =========================================================================
    # OPTIMIZE
    #
    # Description : Optimizes the graph starting from the current estimate.
    # Output      : The optimized 3xN poses (also stored in thePoses).
    # =========================================================================
    def optimize(self):
        if self.thePoses.shape[1]<2:
            return self.thePoses
        X=self.thePoses.copy()
        theLambda=1e-4
        curCost=self._compute_cost(X)
        for _ in range(self.maxIterations):
            H,b=self._build_system(X)
            numVariables=3*X.shape[1]
            isImproved=False
            # Increase the damping until the cost decreases
            for _ in range(10):
                theStep=spsolve((H+theLambda*identity(numVariables,format='csc')).tocsc(),-b)
                Xnew=X+theStep.reshape(-1,3).T
                Xnew[2]=normalize_angles(Xnew[2])
                newCost=self._compute_cost(Xnew)
                if newCost<curCost:
                    isImproved=True
                    break
                theLambda*=10
            if not isImproved:
                break
            theLambda=max(theLambda/10,1e-9)
            relativeChange=(curCost-newCost)/max(curCost,1e-12)
            X=Xnew
            curCost=newCost
            if relativeChange<1e-7 or np.max(np.abs(theStep))<1e-6:
                break
        self.thePoses=X
        return self.thePoses

    # =========================================================================
    # _BUILD_SYSTEM
    #
    # Description : Builds the normal equations H dx=-b.
    # =========================================================================
    def _build_system(self,X):
        numNodes=X.shape[1]
        theRows=[]
        theCols=[]
        theValues=[]
        b=np.zeros(3*numNodes)
        for i,j,theMeasurement,theInformation in self.theEdges:
            theError,A,B=self._edge_jacobians(X[:,i],X[:,j],theMeasurement)
            blockIndices=[(i,i,A.T@theInformation@A),(i,j,A.T@theInformation@B),(j,i,B.T@theInformation@A),(j,j,B.T@theInformation@B)]
            for r,c,theBlock in blockIndices:
                theRows.append(np.repeat(np.arange(3*r,3*r+3),3))
                theCols.append(np.tile(np.arange(3*c,3*c+3),3))
                theValues.append(theBlock.ravel())
            b[3*i:3*i+3]+=A.T@theInformation@theError
            b[3*j:3*j+3]+=B.T@theInformation@theError
        # Anchor the first node
        theRows.append(np.arange(3))
        theCols.append(np.arange(3))
        theValues.append(np.full(3,self.ANCHOR_INFORMATION))
        H=coo_matrix((np.concatenate(theValues),(np.concatenate(theRows),np.concatenate(theCols))),shape=(3*numNodes,3*numNodes)).tocsc()
        return H,b

    # =========================================================================
    # _COMPUTE_COST
    #
    # Description : Total chi-square error, including the anchor.
    # =========================================================================
    def _compute_cost(self,X):
        theCost=0.0
        for i,j,theMeasurement,theInformation in self.theEdges:
            theError=self._edge_error(X[:,i],X[:,j],theMeasurement)
            theCost+=float(theError@theInformation@theError)
        theAnchor=X[:,0]-self.thePoses[:,0]
        return theCost+self.ANCHOR_INFORMATION*float(theAnchor@theAnchor)

    # =========================================================================
    # _EDGE_ERROR / _EDGE_JACOBIANS
    #
    # Description : Error of an edge and its Jacobians with respect to the
    #               two involved poses.
    # =========================================================================
    @staticmethod
    def _edge_error(Xi,Xj,Z):
        theError=relative_motion(Z,relative_motion(Xi,Xj))
        theError[2]=normalize_angles(theError[2])
        return theError

    @staticmethod
    def _edge_jacobians(Xi,Xj,Z):
        ci,si=np.cos(Xi[2]),np.sin(Xi[2])
        cz,sz=np.cos(Z[2]),np.sin(Z[2])
        RiT=np.array([[ci,si],[-si,ci]])
        RzT=np.array([[cz,sz],[-sz,cz]])
        dRiT=np.array([[-si,ci],[-ci,-si]])
        dt=Xj[:2]-Xi[:2]
        theError=np.zeros(3)
        theError[:2]=RzT@(RiT@dt-Z[:2])
        theError[2]=normalize_angles(Xj[2]-Xi[2]-Z[2])
        A=np.zeros((3,3))
        A[:2,:2]=-RzT@RiT
        A[:2,2]=RzT@dRiT@dt
        A[2,2]=-1
        B=np.zeros((3,3))
        B[:2,:2]=RzT@RiT
        B[2,2]=1
        return theError,A,B
