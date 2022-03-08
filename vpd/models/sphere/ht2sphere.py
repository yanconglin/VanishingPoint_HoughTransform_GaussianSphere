import os
import math
import numpy as np
import warnings
from glob import glob

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

def load_cpp_ext(ext_name):
    root_dir = os.path.join(os.path.split(__file__)[0])
    src_dir = os.path.join(root_dir, "cpp_ht2sphere")
    tar_dir = os.path.join(src_dir, "build", ext_name)
    os.makedirs(tar_dir, exist_ok=True)
    srcs = glob(f"{src_dir}/*.cu") + glob(f"{src_dir}/*.cpp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from torch.utils.cpp_extension import load
        ext = load(
            name=ext_name,
            sources=srcs,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[],
            build_directory=tar_dir,
        )
    return ext

# defer calling load_cpp_ext to make CUDA_VISIBLE_DEVICES happy
ht2sphere = None


class HT2SPHERE_Function(Function):
    @staticmethod
    def forward(ctx,
                input_ht,
                sphere_index,
                height, width,
                sphere_size
                ):
        ctx.height =height
        ctx.width =width
        ctx.sphere_size =sphere_size
        ctx.save_for_backward(sphere_index)
        output = ht2sphere.ht2sphere_forward(
            input_ht,
            sphere_index,
            ctx.height,
            ctx.width,
            ctx.sphere_size
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        sphere_index = ctx.saved_tensors[0]  # it is a list of length 1!

        grad_input = ht2sphere.ht2sphere_backward(
            grad_output,
            sphere_index,
            ctx.height,
            ctx.width,
            ctx.sphere_size
        )

        return ( 
            grad_input,  # input_ht, forward pass has three inputs
            None, # sphere_index
            None, # height
            None, # width
            None, # sphere_size
        )


class HT2SPHERE(nn.Module):
    def __init__(self, ht_size, sphere_size, vote_mapping):
        super(HT2SPHERE, self).__init__()
        
        vote_mapping.requires_grad=False
        self.register_buffer('vote_mapping', vote_mapping, persistent=False)
        
        global ht2sphere
        print('#################### ht2sphere compiling ############################')
        ht2sphere = load_cpp_ext("ht2sphere")
        print('#################### done! ############################')

        self.sphere_size = sphere_size
        self.ht_size = _pair(ht_size)
        self.num_votes, _ = self.vote_mapping.shape
        
        assert self.vote_mapping[:,2].max().item() < self.sphere_size, "vote_index max ind >= sphere_size"
        self.__repr__()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'ht_size=' + str(self.ht_size) \
               + ', sphere_size=' + str(self.sphere_size)\
               + ', num_votes=' + str(self.num_votes) +  ')'


    def forward(self, input):  

        batch, channel, h, w = input.size()
        out = HT2SPHERE_Function.apply(
            input.contiguous(),
            self.vote_mapping,
            self.ht_size[0],
            self.ht_size[1],
            self.sphere_size
        )
        return out.view(batch, channel, self.sphere_size)

