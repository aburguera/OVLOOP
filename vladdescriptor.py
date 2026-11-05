#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : VLADDescriptor
# Description : Vector of Locally Aggregated Descriptors (Jegou et al., 2010)
#               built on SIFT local features, with intra-normalization,
#               signed square root (power) normalization and a final L2
#               normalization. The similarity of two descriptors is their
#               dot product (cosine similarity).
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class VLADDescriptorError(Exception):
    pass

###############################################################################
# CLASS
###############################################################################
class VLADDescriptor:

    # CLASS CONSTANTS
    NUM_FEATURES=500            # SIFT features per image.

    # =========================================================================
    # CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT LOAD NOR BUILD ANYTHING.
    # Input  : vocabularySize - Number of visual words (used by create).
    # =========================================================================
    def __init__(self,vocabularySize=32):
        self.vocabularySize=vocabularySize
        self.theVocabulary=None
        self._theSIFT=None

    # =========================================================================
    # CREATE
    #
    # Description : Builds the vocabulary by k-means clustering of the SIFT
    #               descriptors of a set of training images. Not needed if a
    #               vocabulary is loaded.
    # Input       : trainImages - Iterable of HxWx3 uint8 RGB images.
    #               randomSeed - Seed of the k-means.
    # =========================================================================
    def create(self,trainImages,randomSeed=0):
        self._theSIFT=cv2.SIFT_create(nfeatures=self.NUM_FEATURES)
        allDescriptors=np.vstack([self._extract(curImage) for curImage in trainImages])
        theKMeans=MiniBatchKMeans(n_clusters=self.vocabularySize,random_state=randomSeed,batch_size=4096,n_init=3)
        theKMeans.fit(allDescriptors)
        self.theVocabulary=theKMeans.cluster_centers_.astype(np.float32)

    # =========================================================================
    # SAVE / LOAD
    #
    # Description : Stores or retrieves the vocabulary.
    # Input       : fileName - Path of the .npy vocabulary file.
    # =========================================================================
    def save(self,fileName):
        np.save(fileName,self.theVocabulary)

    def load(self,fileName):
        self.theVocabulary=np.load(fileName).astype(np.float32)
        self.vocabularySize=self.theVocabulary.shape[0]
        self._theSIFT=cv2.SIFT_create(nfeatures=self.NUM_FEATURES)

    # =========================================================================
    # COMPUTE
    #
    # Description : VLAD descriptor of one image.
    # Input       : theImage - HxWx3 uint8 RGB image.
    # Output      : L2-normalized float32 vector of vocabularySize*128 values.
    # =========================================================================
    def compute(self,theImage):
        if self.theVocabulary is None:
            raise VLADDescriptorError('The vocabulary must be created or loaded first')
        localDescriptors=self._extract(theImage)
        theVLAD=np.zeros((self.vocabularySize,128),dtype=np.float32)
        if localDescriptors.shape[0]>0:
            theWords=self._assign_words(localDescriptors)
            np.add.at(theVLAD,theWords,localDescriptors-self.theVocabulary[theWords])
        # Intra-normalization, power normalization and L2 normalization
        theVLAD/=np.maximum(np.linalg.norm(theVLAD,axis=1,keepdims=True),1e-12)
        theVLAD=np.sign(theVLAD)*np.sqrt(np.abs(theVLAD))
        theVLAD=theVLAD.ravel()
        return (theVLAD/max(np.linalg.norm(theVLAD),1e-12)).astype(np.float32)

    # =========================================================================
    # _EXTRACT
    #
    # Description : L2-normalized SIFT descriptors of an image.
    # =========================================================================
    def _extract(self,theImage):
        grayImage=cv2.cvtColor(theImage,cv2.COLOR_RGB2GRAY)
        _,theDescriptors=self._theSIFT.detectAndCompute(grayImage,None)
        if theDescriptors is None:
            return np.zeros((0,128),dtype=np.float32)
        theDescriptors=theDescriptors/np.maximum(np.linalg.norm(theDescriptors,axis=1,keepdims=True),1e-12)
        return theDescriptors.astype(np.float32)

    # =========================================================================
    # _ASSIGN_WORDS
    #
    # Description : Nearest visual word of each local descriptor.
    # =========================================================================
    def _assign_words(self,localDescriptors):
        theDistances=np.sum(localDescriptors**2,axis=1,keepdims=True)-2*localDescriptors@self.theVocabulary.T+np.sum(self.theVocabulary**2,axis=1)[None,:]
        return np.argmin(theDistances,axis=1)
