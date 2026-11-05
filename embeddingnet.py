#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : EmbeddingNet
# Description : FaceNet-style image embedding network for loop retrieval with
#               bottom-looking underwater cameras. A self-supervised DINOv2
#               ViT-S/14 backbone (Oquab et al., 2024) provides a class token
#               and a set of patch tokens. The class token and the mean of the
#               patch tokens are concatenated, linearly projected and L2 nor-
#               malized. Thus, embeddings lie on the unit hypersphere and the
#               cosine similarity of two embeddings is in [-1,1]. Only the last
#               blocks of the backbone and the projection are trained; the
#               rest of the backbone keeps the official DINOv2 weights.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 08-Oct-2026 - Creation.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# CLASS
###############################################################################
class EmbeddingNet(nn.Module):

    # CLASS CONSTANTS
    IMAGENET_MEAN=(0.485,0.456,0.406)   # Input normalization (ImageNet).
    IMAGENET_STD=(0.229,0.224,0.225)

    # =========================================================================
    # CONSTRUCTOR. PLEASE NOTE THAT THIS DOWNLOADS THE OFFICIAL DINOV2 WEIGHTS
    # FROM PYTORCH HUB THE FIRST TIME IT IS USED.
    # Input  : embeddingSize - Dimension of the output embedding.
    #          inputSize - (rows,cols) of the network input. Images of any
    #                      size are resized to it.
    #          numTrainableBlocks - Number of trainable transformer blocks.
    # =========================================================================
    def __init__(self,embeddingSize=128,inputSize=(224,294),numTrainableBlocks=4):
        super().__init__()
        self.inputSize=tuple(inputSize)
        self.register_buffer('theMean',torch.tensor(self.IMAGENET_MEAN).view(1,3,1,1))
        self.register_buffer('theStd',torch.tensor(self.IMAGENET_STD).view(1,3,1,1))
        self.theBackbone=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',pretrained=True,trust_repo=True)
        # Freeze the backbone except its last blocks and its final norm
        for curParameter in self.theBackbone.parameters():
            curParameter.requires_grad=False
        for curBlock in self.theBackbone.blocks[-numTrainableBlocks:]:
            for curParameter in curBlock.parameters():
                curParameter.requires_grad=True
        for curParameter in self.theBackbone.norm.parameters():
            curParameter.requires_grad=True
        self.theProjection=nn.Linear(2*self.theBackbone.embed_dim,embeddingSize)

    # =========================================================================
    # FORWARD
    # Input  : theImages - Bx3xHxW float images in [0,1].
    # Output : BxD L2-normalized embeddings.
    # =========================================================================
    def forward(self,theImages):
        if tuple(theImages.shape[2:])!=self.inputSize:
            theImages=F.interpolate(theImages,size=self.inputSize,mode='bilinear',align_corners=False,antialias=True)
        theImages=(theImages-self.theMean)/self.theStd
        theTokens=self.theBackbone.forward_features(theImages)
        thePooled=torch.cat((theTokens['x_norm_clstoken'],theTokens['x_norm_patchtokens'].mean(dim=1)),dim=1)
        return F.normalize(self.theProjection(thePooled),p=2,dim=1)

    # =========================================================================
    # GET_TRAINED_STATE
    #
    # Description : State of the trained parameters only (trainable backbone
    #               blocks and projection). The frozen part of the backbone
    #               is not stored, since it is the official DINOv2 one.
    # Output      : Dict parameter name -> tensor.
    # =========================================================================
    def get_trained_state(self):
        trainedNames={n for n,p in self.named_parameters() if p.requires_grad}
        return {n:v.detach().cpu().clone() for n,v in self.state_dict().items() if n in trainedNames}

    # =========================================================================
    # LOAD_TRAINED_STATE
    #
    # Description : Loads the trained parameters stored by get_trained_state.
    # Input       : theState - Dict parameter name -> tensor.
    # =========================================================================
    def load_trained_state(self,theState):
        theResult=self.load_state_dict(theState,strict=False)
        if len(theResult.unexpected_keys)>0:
            raise ValueError('Unexpected parameters in the stored state: %s'%', '.join(theResult.unexpected_keys))
