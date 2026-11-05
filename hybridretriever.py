#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : HybridRetriever
# Description : Hybrid loop retrieval (HYB). Each image is described by two
#               global descriptors: a hand-crafted VLAD descriptor and a
#               learned embedding (EmbeddingNet). Both are very precise in
#               different situations: VLAD when the fine texture of the sea
#               floor is preserved and the embedding under turbidity. Their
#               cosine similarities are standardized (using the mean and the
#               standard deviation measured on validation data) and fused:
#                   s = w*(sV-muV)/sigmaV + (1-w)*(sE-muE)/sigmaE
#               The provided embedding network and statistics were obtained
#               from images simulated over photomosaics. With real images,
#               it may be necessary to retrain the network or, at least, to
#               recompute the statistics.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import os
import json
import numpy as np
import torch
from vladdescriptor import VLADDescriptor
from embeddingnet import EmbeddingNet

###############################################################################
# CLASS
###############################################################################
class HybridRetriever:

    # CLASS CONSTANTS
    VOCABULARY_FILE='vlad_vocabulary.npy'       # VLAD vocabulary.
    WEIGHTS_FILE='embedding_weights.pt'         # Trained embedding parameters.
    FUSION_FILE='hybrid_retrieval.json'         # Fusion weight and statistics.

    # =========================================================================
    # CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT LOAD THE MODELS.
    # Input  : theDevice - Torch device name ('cpu', 'cuda', 'mps') or None
    #                      to choose it automatically.
    # =========================================================================
    def __init__(self,theDevice=None):
        if theDevice is None:
            theDevice='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.theDevice=torch.device(theDevice)
        self.theVLAD=None
        self.theNet=None
        self.fusionParams=None

    # =========================================================================
    # LOAD
    #
    # Description : Loads the VLAD vocabulary, the embedding network and the
    #               fusion parameters.
    # Input       : modelsPath - Folder with the model files.
    # =========================================================================
    def load(self,modelsPath):
        self.theVLAD=VLADDescriptor()
        self.theVLAD.load(os.path.join(modelsPath,self.VOCABULARY_FILE))
        self.theNet=EmbeddingNet()
        self.theNet.load_trained_state(torch.load(os.path.join(modelsPath,self.WEIGHTS_FILE),map_location='cpu'))
        self.theNet=self.theNet.to(self.theDevice).eval()
        with open(os.path.join(modelsPath,self.FUSION_FILE),'r') as inFile:
            self.fusionParams=json.load(inFile)

    # =========================================================================
    # COMPUTE
    #
    # Description : Global descriptors of one image.
    # Input       : theImage - HxWx3 uint8 RGB image.
    # Output      : theDescriptor - Dict with 'vlad' and 'embedding' vectors.
    # =========================================================================
    def compute(self,theImage):
        theTensor=torch.from_numpy(np.ascontiguousarray(theImage)).permute(2,0,1).unsqueeze(0).float().to(self.theDevice)/255.0
        with torch.no_grad():
            theEmbedding=self.theNet(theTensor).cpu().numpy()[0]
        return {'vlad':self.theVLAD.compute(theImage),'embedding':theEmbedding}

    # =========================================================================
    # SIMILARITY
    #
    # Description : Hybrid similarity between a query and a set of images.
    # Input       : queryDescriptor - Dict returned by compute.
    #               vladDatabase - NxDv VLAD descriptors of the images.
    #               embeddingDatabase - NxDe embeddings of the images.
    # Output      : Array of N hybrid similarities (larger is more similar).
    # =========================================================================
    def similarity(self,queryDescriptor,vladDatabase,embeddingDatabase):
        fp=self.fusionParams
        vladSimilarity=(vladDatabase@queryDescriptor['vlad']-fp['vladMean'])/fp['vladStd']
        embeddingSimilarity=(embeddingDatabase@queryDescriptor['embedding']-fp['embeddingMean'])/fp['embeddingStd']
        return fp['vladWeight']*vladSimilarity+(1-fp['vladWeight'])*embeddingSimilarity
