import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
import scipy.io as sio
import random
import time
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import time
import os
from .ht2sphere import HT2SPHERE

class SPHERE_CUDA(nn.Module):

    def __init__(self, vote_mapping_dict):
        super(SPHERE_CUDA, self).__init__()
        vote_mapping=vote_mapping_dict["vote_mapping"]
        self.register_buffer("vote_mapping", vote_mapping, persistent=False)
        self.ht_size = _pair(vote_mapping_dict["ht_size"])
        self.sphere_size = vote_mapping_dict["sphere_size"]
        self.num_votes = vote_mapping.shape[1]
        # print(vote_mapping.shape, vote_mapping[:,0].max().item() )
        assert len(vote_mapping) == self.sphere_size, "vote_mapping max ind >= sphere_size"
        assert vote_mapping[:,:, 0].max().item() < self.ht_size[0]*self.ht_size[1], "vote_mapping max ind >= ht_size"
        
        self.sphere = HT2SPHERE().double()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'ht_size=' + str(self.ht_size) \
               + ', sphere_size=' + str(self.sphere_size) + ')'
        
    def forward(self, x, inds):
        B, C, H, W = x.shape

        with torch.no_grad():
            num_pts = inds.shape[1]
            mapping = torch.index_select(self.vote_mapping, dim=0, index=inds.flatten()) # []
            mapping = mapping.view(B, num_pts, self.num_votes, 2) # [B, num_pts, num_votes, 2]


        out = self.sphere(x, mapping)

        return out
