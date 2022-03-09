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
                ):
        _, _, height, width = input_ht.shape
        ctx.height =height
        ctx.width =width
        _, num_pts, num_votes, _ = sphere_index.shape
        ctx.num_pts =num_pts
        ctx.num_votes =num_votes

        ctx.save_for_backward(sphere_index)
        output = ht2sphere.ht2sphere_forward(
            input_ht,
            sphere_index,
            ctx.height,
            ctx.width,
            ctx.num_pts,
            ctx.num_votes
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
            ctx.num_pts,
            ctx.num_votes
        )

        return ( 
            grad_input,  # input_ht, forward pass has three inputs
            None, # sphere_index
            None, # height
            None, # width
            None, # num_pts
            None  # num_votes
        )


class HT2SPHERE(nn.Module):
    def __init__(self):
        super(HT2SPHERE, self).__init__()
        
        global ht2sphere
        print('#################### ht2sphere compiling ############################')
        ht2sphere = load_cpp_ext("ht2sphere")
        print('#################### ht2sphere compiling done! ############################')

        self.__repr__()


    def forward(self, feats, mapping):  

        B,C,H,W = feats.size()
        _, num_pts, num_votes, _ = mapping.size()
        out =  HT2SPHERE_Function.apply(
            feats.contiguous(),
            mapping.contiguous(),
        )
        return out

