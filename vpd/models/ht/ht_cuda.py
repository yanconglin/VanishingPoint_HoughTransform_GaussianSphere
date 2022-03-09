import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
from .im2ht import IM2HT

class HT_CUDA(nn.Module):
    """ mapping from pixels to HT
        input: image feaures [b, c, r, c]
        output: HT feaures [b, c, h, w]
        precalulation: vote_mapping [?, 3]
        [:,0]: pixel index
        [:,1]: HT bin index
        [:,2]: weight estimated from quantization

    """
    def __init__(self, vote_mapping_dict):
        super(HT_CUDA, self).__init__()
        self.im_size = _pair(vote_mapping_dict["im_size"])
        self.ht_size = _pair(vote_mapping_dict["ht_size"])
        self.norm = float(max(self.im_size))

        vote_mapping=vote_mapping_dict["vote_mapping"]
        
        assert vote_mapping[:,0].max().item() < self.im_size[0]*self.im_size[1], "vote_mapping max ind >= im_size"
        assert vote_mapping[:,1].max().item() < self.ht_size[0]*self.ht_size[1], "vote_mapping max ind >= ht_size"
        
        self.ht = IM2HT(im_size=self.im_size, ht_size=self.ht_size, vote_mapping=vote_mapping)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'im_size=' + str(self.im_size) + ', ht_size=' + str(self.ht_size) + ')'

    # 128x128:
    # def forward(self, x):
    #     out = self.ht(x)
    #     # print("ht_cuda", out.shape)
    #     return out / self.norm

    # 256x256: there might be an error: launching too many cores; a naive way to solve
    def forward(self, x):
        batch, channel, rows, cols = x.shape
        assert channel%2==0
        x = x.view(batch*2, channel//2, rows, cols)
        out = self.ht(x)
        out = out.view(batch, channel, self.ht_size[0], self.ht_size[1])
        return out / self.norm
