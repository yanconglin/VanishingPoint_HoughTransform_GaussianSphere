import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
import scipy.io as sio
import random
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import time
import os
from .ht2im import HT2IM

class IHT_CUDA(nn.Module):

    def __init__(self, vote_mapping_dict):
        super(IHT_CUDA, self).__init__()
        self.im_size = _pair(vote_mapping_dict["im_size"])
        self.ht_size = _pair(vote_mapping_dict["ht_size"])
        self.norm = float(max(self.im_size))

        vote_mapping=vote_mapping_dict["vote_mapping"]
        assert vote_mapping[:,0].max().item() < self.im_size[0]*self.im_size[1], "vote_mapping max ind >= im_size"
        assert vote_mapping[:,1].max().item() < self.ht_size[0]*self.ht_size[1], "vote_mapping max ind >= ht_size"
        
        self.iht = HT2IM(im_size=self.im_size, ht_size=self.ht_size, vote_mapping=vote_mapping)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'im_size=' + str(self.im_size) + ', ht_size=' + str(self.ht_size) + ')'

    def forward(self, x):
        out = self.iht(x)
        # print("iht_cuda", out.shape)
        return out

    
    # def forward(self, x):
    #     batch, channel, h, w = x.shape
    #     assert channel%2==0
    #     x = x.view(batch*2, channel//2, h, w)
    #     out = self.iht(x)
    #     out = out.view(batch, channel, self.im_size[0], self.im_size[1])
    #     # print("iht_cuda", out.shape)
    #     return out
