#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : SLAMSystem
# Description : On-line 2D visual Graph SLAM for an AUV endowed with a
#               bottom-looking camera, an altimeter and an odometer. For
#               every new image:
#                 1) A node is added to the pose graph using the odometry.
#                 2) Loop candidates are retrieved among the past images
#                    (excluding the most recent ones) by means of the hybrid
#                    retrieval (HybridRetriever).
#                 3) Each candidate is geometrically verified with a metric
#                    rigid model (ImageRegistrator).
#                 4) The verified candidates are validated by Query-level
#                    Consensus Validation with Gating (QCVG):
#                    * INL: enough RANSAC inliers.
#                    * GATE: the loop must agree with the prediction of the
#                      current map (squared Mahalanobis distance below the
#                      calibrated threshold). The uncertainty of the
#                      prediction is propagated along the least uncertain
#                      path of the current graph.
#                    * QCV: two candidates a and b of the same query are
#                      consistent if the motion from a to b implied through
#                      the query agrees with the motion from a to b according
#                      to the map. The largest set of mutually consistent
#                      candidates (maximum clique) is accepted if it is
#                      unique and has at least two members. Ambiguous
#                      queries are rejected. A single candidate that passes
#                      INL and GATE is accepted.
#                 5) The accepted loops are added to the graph, which is
#                    optimized.
#               The estimated poses are those of the camera, x and y being
#               aligned with the image columns and rows. All the distances
#               are expressed in world units (those of the altitude and the
#               odometry).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import heapq
import numpy as np
from posegraph import PoseGraph
from transform2d import compose_references,relative_motion,normalize_angles,compose_with_covariance,invert_with_covariance

class SLAMSystemError(Exception):
    pass

###############################################################################
# CLASS
###############################################################################
class SLAMSystem:

    # =========================================================================
    # CONSTRUCTOR
    # Input  : slamParams - SLAMParameters object.
    #          theRetriever - Loaded HybridRetriever.
    #          theRegistrator - ImageRegistrator.
    #          initialPose - Pose [x,y,o] of the first image.
    # =========================================================================
    def __init__(self,slamParams,theRetriever,theRegistrator,initialPose=(0.0,0.0,0.0)):
        self.slamParams=slamParams
        self.theRetriever=theRetriever
        self.theRegistrator=theRegistrator
        self.theGraph=PoseGraph()
        self.theGraph.add_node(np.asarray(initialPose,dtype=float))
        self.loopCovariance=np.diag([slamParams.loopSigmaXY**2,slamParams.loopSigmaXY**2,slamParams.loopSigmaTheta**2])
        self.loopInformation=np.linalg.inv(self.loopCovariance)
        self.theAdjacency=[[]]
        self.acceptedLoops=[]
        self.vladDatabase=[]
        self.embeddingDatabase=[]
        self.theFeatures=[]
        self.theAltitudes=[]
        self.lastOptimization=0
        self.hasNewLoops=False
        self._theInput=None

    # =========================================================================
    # SET_INPUT
    #
    # Description : Defines the data of the next image to process.
    # Input       : theImage - HxWx3 uint8 RGB image.
    #               theAltitude - Altimeter reading (world units).
    #               theOdometry - Motion [x,y,o] from the previous image to
    #                   this one, in world units, expressed in the camera
    #                   frame of the previous image (x along the image col-
    #                   umns, y along the image rows). This class does not
    #                   compute the odometry: it has to be provided by the
    #                   navigation of the vehicle (or by the DataSimulator),
    #                   transformed to the camera frame if needed. Ignored
    #                   for the first image.
    #               odoCovariance - 3x3 covariance of theOdometry. Ignored
    #                   for the first image.
    # =========================================================================
    def set_input(self,theImage,theAltitude,theOdometry=None,odoCovariance=None):
        if len(self.theFeatures)>0 and (theOdometry is None or odoCovariance is None):
            raise SLAMSystemError('Odometry and its covariance are required for all the images but the first one')
        self._theInput=(theImage,float(theAltitude),theOdometry,odoCovariance)

    # =========================================================================
    # PROCESS
    #
    # Description : Processes the image defined by set_input.
    # =========================================================================
    def process(self):
        if self._theInput is None:
            raise SLAMSystemError('No input. Call set_input first')
        theImage,theAltitude,theOdometry,odoCovariance=self._theInput
        self._theInput=None
        sp=self.slamParams

        # Store the image description
        theDescriptor=self.theRetriever.compute(theImage)
        self.vladDatabase.append(theDescriptor['vlad'])
        self.embeddingDatabase.append(theDescriptor['embedding'])
        theFeatures=self.theRegistrator.extract(theImage)
        theFeatures['descriptors']=theFeatures['descriptors'].astype(np.uint8)
        self.theFeatures.append(theFeatures)
        self.theAltitudes.append(theAltitude)
        t=len(self.theFeatures)-1
        if t==0:
            return

        # Odometry
        newPose=compose_references(self.theGraph.thePoses[:,-1],np.asarray(theOdometry,dtype=float))
        self.theGraph.add_node(newPose)
        self.theGraph.add_edge(t-1,t,theOdometry,np.linalg.inv(odoCovariance))
        self.theAdjacency.append([])
        self._add_adjacency(t-1,t,np.asarray(theOdometry,dtype=float),np.asarray(odoCovariance,dtype=float))
        if t<=sp.excludeWindow:
            return

        # Loop candidate retrieval
        theSimilarities=self.theRetriever.similarity(theDescriptor,np.array(self.vladDatabase[:t-sp.excludeWindow]),np.array(self.embeddingDatabase[:t-sp.excludeWindow]))
        theCandidates=np.argsort(-theSimilarities)[:sp.numCandidates]

        # Geometric verification
        theResults=[self._register(c,t) for c in theCandidates]

        # Validation (QCVG): inliers and map consistency, then consensus
        mapDistances=self._map_distances(t,theCandidates,theResults)
        isAccepted=np.array([r['isValid'] and r['numInliers']>=sp.inlierThreshold and d<=sp.mapThreshold for r,d in zip(theResults,mapDistances)],dtype=bool)
        isAccepted=self._query_consensus(theCandidates,theResults,isAccepted)
        for c,r,a in zip(theCandidates,theResults,isAccepted):
            if a:
                self.theGraph.add_edge(c,t,r['theMotion'],self.loopInformation)
                self._add_adjacency(c,t,r['theMotion'],self.loopCovariance)
                self.acceptedLoops.append((int(c),int(t)))
                self.hasNewLoops=True

        # Graph optimization
        if self.hasNewLoops and t-self.lastOptimization>=sp.optimizeEvery:
            self.theGraph.optimize()
            self.lastOptimization=t
            self.hasNewLoops=False

    # =========================================================================
    # OPTIMIZE
    #
    # Description : Optimizes the graph. Call it at the end of the mission.
    # =========================================================================
    def optimize(self):
        self.theGraph.optimize()
        self.hasNewLoops=False

    # =========================================================================
    # GET_POSES / GET_LOOPS
    #
    # Description : Current estimate and accepted loops.
    # Output      : 3xN poses / list of (candidate,query) image indices.
    # =========================================================================
    def get_poses(self):
        return self.theGraph.thePoses.copy()

    def get_loops(self):
        return list(self.acceptedLoops)

    # =========================================================================
    # _REGISTER
    #
    # Description : Registration of image t (current) against image c
    #               (reference). The motion is from camera c to camera t.
    # =========================================================================
    def _register(self,c,t):
        refFeatures=dict(self.theFeatures[c],descriptors=self.theFeatures[c]['descriptors'].astype(np.float32))
        curFeatures=dict(self.theFeatures[t],descriptors=self.theFeatures[t]['descriptors'].astype(np.float32))
        return self.theRegistrator.register(refFeatures,curFeatures,self.theAltitudes[c],self.theAltitudes[t])

    # =========================================================================
    # _ADD_ADJACENCY
    #
    # Description : Stores an edge (both directions) for shortest path
    #               searches. The cost of an edge is the trace of its trans-
    #               lational covariance.
    # =========================================================================
    def _add_adjacency(self,i,j,theMotion,theCovariance):
        theCost=float(theCovariance[0,0]+theCovariance[1,1])
        self.theAdjacency[i].append((j,theCost,theMotion,theCovariance,False))
        self.theAdjacency[j].append((i,theCost,theMotion,theCovariance,True))

    # =========================================================================
    # _PATH_MOTION
    #
    # Description : Motion and covariance obtained by composing a path.
    # Input       : thePath - List of (motion,covariance,isReversed) edges.
    # Output      : pathMotion,pathCovariance - Composed motion and its
    #                   covariance.
    # =========================================================================
    @staticmethod
    def _path_motion(thePath):
        pathMotion=np.zeros(3)
        pathCovariance=np.zeros((3,3))
        for theMotion,theCovariance,isReversed in thePath:
            if isReversed:
                theMotion,theCovariance=invert_with_covariance(theMotion,theCovariance)
            pathMotion,pathCovariance=compose_with_covariance(pathMotion,pathCovariance,theMotion,theCovariance)
        return pathMotion,pathCovariance

    # =========================================================================
    # _MAP_DISTANCES
    #
    # Description : Squared Mahalanobis distance between each candidate loop
    #               motion and the motion predicted by the current graph.
    #               The covariance of the prediction is obtained by compo-
    #               sing the edge covariances along the shortest (least
    #               uncertain) path from the candidate to the query.
    # Input       : t - Query index.
    #               theCandidates - Candidate indices.
    #               theResults - Registration results.
    # Output      : Array of squared distances (inf if the model is invalid).
    # =========================================================================
    def _map_distances(self,t,theCandidates,theResults):
        thePaths=self._shortest_paths(t,{int(c) for c in theCandidates})
        theDistances=np.full(len(theCandidates),np.inf)
        for k,(c,r) in enumerate(zip(theCandidates,theResults)):
            if not r['isValid'] or int(c) not in thePaths:
                continue
            _,pathCovariance=self._path_motion(thePaths[int(c)])
            predictedMotion=relative_motion(self.theGraph.thePoses[:,c],self.theGraph.thePoses[:,t])
            theError=r['theMotion']-predictedMotion
            theError[2]=normalize_angles(theError[2])
            theDistances[k]=float(theError@np.linalg.solve(pathCovariance+self.loopCovariance,theError))
        return theDistances

    # =========================================================================
    # _QUERY_CONSENSUS
    #
    # Description : Query-level consensus. Every verified candidate a of the
    #               query t provides a motion z_a (a->t). Two candidates a,b
    #               are consistent if the motion a->b implied through the
    #               query, z_a (+) (-)z_b, agrees with the motion a->b accor-
    #               ding to the map, whose uncertainty is propagated along the
    #               shortest graph path between the two old nodes a and b (so
    #               the drift accumulated by the query is not involved). The
    #               largest set of mutually consistent candidates is accepted
    #               if it is unique and has at least two members; otherwise
    #               (ambiguous query) all the candidates are rejected. Queries
    #               with a single verified candidate are kept unchanged.
    # Input       : theCandidates - Candidate indices.
    #               theResults - Registration results.
    #               isAccepted - Acceptance after the INL and GATE tests.
    # Output      : Updated acceptance.
    # =========================================================================
    def _query_consensus(self,theCandidates,theResults,isAccepted):
        theVerified=np.nonzero(isAccepted)[0]
        numVerified=theVerified.size
        if numVerified<2:
            return isAccepted
        isConsistent=np.eye(numVerified,dtype=bool)
        for p in range(numVerified):
            a=int(theCandidates[theVerified[p]])
            otherNodes={int(theCandidates[theVerified[q]]) for q in range(p+1,numVerified)}
            thePaths=self._shortest_paths(a,otherNodes) if len(otherNodes)>0 else {}
            for q in range(p+1,numVerified):
                b=int(theCandidates[theVerified[q]])
                if a==b:
                    isConsistent[p,q]=isConsistent[q,p]=True
                    continue
                if b not in thePaths:
                    continue
                # Map motion b->a along the path, then its inverse a->b
                pathMotion,pathCovariance=self._path_motion(thePaths[b])
                _,pathCovariance=invert_with_covariance(pathMotion,pathCovariance)
                predictedMotion=relative_motion(self.theGraph.thePoses[:,a],self.theGraph.thePoses[:,b])
                # Motion a->b implied through the query
                firstMotion=theResults[theVerified[p]]['theMotion']
                secondMotion=theResults[theVerified[q]]['theMotion']
                impliedMotion,impliedCovariance=compose_with_covariance(firstMotion,self.loopCovariance,*invert_with_covariance(secondMotion,self.loopCovariance))
                theError=impliedMotion-predictedMotion
                theError[2]=normalize_angles(theError[2])
                theDistance=float(theError@np.linalg.solve(pathCovariance+impliedCovariance,theError))
                isConsistent[p,q]=isConsistent[q,p]=theDistance<=self.slamParams.mapThreshold
        bestSets=self._maximum_cliques(isConsistent)
        isAccepted=np.zeros_like(isAccepted)
        if len(bestSets)==1 and len(bestSets[0])>=2:
            isAccepted[theVerified[list(bestSets[0])]]=True
        return isAccepted

    # =========================================================================
    # _MAXIMUM_CLIQUES
    #
    # Description : All the maximum cliques of a small consistency graph
    #               (exhaustive search; the graph has at most numCandidates
    #               nodes).
    # Input       : isConsistent - Symmetric boolean adjacency matrix.
    # Output      : List of maximum cliques (tuples of node indices).
    # =========================================================================
    @staticmethod
    def _maximum_cliques(isConsistent):
        numNodes=isConsistent.shape[0]
        bestSets=[]
        bestSize=0
        for theMask in range(1,2**numNodes):
            theNodes=[k for k in range(numNodes) if theMask>>k&1]
            if len(theNodes)<bestSize:
                continue
            if not np.all(isConsistent[np.ix_(theNodes,theNodes)]):
                continue
            if len(theNodes)>bestSize:
                bestSize=len(theNodes)
                bestSets=[]
            bestSets.append(tuple(theNodes))
        return bestSets

    # =========================================================================
    # _SHORTEST_PATHS
    #
    # Description : Dijkstra search from a source node over the current
    #               graph, stopping when all the targets are reached.
    # Input       : t - Source node.
    #               theTargets - Set of target nodes.
    # Output      : Dict target -> list of (motion,covariance,isReversed)
    #               edges ordered from the target to the source.
    # =========================================================================
    def _shortest_paths(self,t,theTargets):
        theCosts={t:0.0}
        cameFrom={t:None}
        theHeap=[(0.0,t)]
        pendingTargets=set(theTargets)
        while theHeap and pendingTargets:
            curCost,curNode=heapq.heappop(theHeap)
            if curCost>theCosts[curNode]:
                continue
            pendingTargets.discard(curNode)
            for nextNode,edgeCost,theMotion,theCovariance,isReversed in self.theAdjacency[curNode]:
                newCost=curCost+edgeCost
                if newCost<theCosts.get(nextNode,np.inf):
                    theCosts[nextNode]=newCost
                    # Edge traversed from nextNode to curNode. The stored
                    # motion goes from the lower to the higher node unless
                    # it is reversed.
                    cameFrom[nextNode]=(curNode,theMotion,theCovariance,not isReversed)
                    heapq.heappush(theHeap,(newCost,nextNode))
        thePaths={}
        for curTarget in theTargets:
            if curTarget not in cameFrom:
                continue
            theEdges=[]
            curNode=curTarget
            while cameFrom[curNode] is not None:
                parentNode,theMotion,theCovariance,isReversed=cameFrom[curNode]
                theEdges.append((theMotion,theCovariance,isReversed))
                curNode=parentNode
            thePaths[curTarget]=theEdges
        return thePaths
