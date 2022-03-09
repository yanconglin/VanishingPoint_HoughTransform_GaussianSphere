import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair
from .ht2sphere import HT2SPHERE

class SPHERE_CUDA(nn.Module):
    """ mapping from HT to Sphere 
        input: HT feaures [b, c, h, w]
        output: Sphere feaures [b, c, num_points]
        precalulation: vote_mapping [?, 3]
        [:,0]: HT index
        [:,1]: Spherical point index
        [:,2]: weight, cos dis to the nn spherical point (fibonacci lattice)

    """

    def __init__(self, vote_mapping_dict):
        super(SPHERE_CUDA, self).__init__()
        vote_mapping=vote_mapping_dict["vote_mapping"]
        self.num_votes, _ = vote_mapping.shape
        self.ht_size = _pair(vote_mapping_dict["ht_size"])
        self.sphere_size = vote_mapping_dict["sphere_size"]
        
        assert vote_mapping[:,0].max().item() < self.ht_size[0]*self.ht_size[1], "vote_mapping max ind >= ht_size"
        assert vote_mapping[:,1].max().item() < self.sphere_size, "vote_mapping max ind >= sphere_size"
        
        
        self.sphere = HT2SPHERE(ht_size=self.ht_size, sphere_size=self.sphere_size, vote_mapping=vote_mapping)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'ht_size=' + str(self.ht_size) \
               + ', sphere_size=' + str(self.sphere_size) + ')'
        
    # def forward(self, x):
    #     out = self.sphere(x)
    #     return out

    # there might be an error for 256x256: launching too many cores; a naive way to solve.
    def forward(self, x):
        batch, channel, h, w = x.shape
        assert channel % 4 == 0
        x = x.view(batch * 4, channel // 4, h, w)
        out = self.sphere(x)
        out = out.view(batch, channel, self.sphere_size)
        return out
