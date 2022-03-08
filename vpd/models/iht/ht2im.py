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
    src_dir = os.path.join(root_dir, "cpp_ht2im")
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
ht2im = None


class HT2IMFunction(Function):
    @staticmethod
    def forward(ctx,
                input_ht,
                ht_index,
                im_size, 
                ht_size):
        ctx.im_size = _pair(im_size)
        ctx.ht_size = _pair(ht_size)
        ctx.save_for_backward(ht_index)
        output = ht2im.ht2im_forward(
            input_ht,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        ht_index = ctx.saved_tensors[0]  # it is a list of length 1!

        grad_input = ht2im.ht2im_backward(
            grad_output,
            ht_index,
            ctx.im_size[0],
            ctx.im_size[1],
            ctx.ht_size[0],
            ctx.ht_size[1]
        )

        return (
            grad_input,  # input
            None, # ht_index
            None, # im_size
            None, # ht_size
        )


class HT2IM(nn.Module):
    def __init__(self, im_size, ht_size, vote_mapping):
        super(HT2IM, self).__init__()
        vote_mapping.requires_grad=False
        self.register_buffer('vote_mapping', vote_mapping, persistent=False)
        global ht2im
        print('#################### ht2im compiling ############################')
        ht2im = load_cpp_ext("ht2im")
        print('#################### done! ############################')

        self.im_size = _pair(im_size)
        self.ht_size = _pair(ht_size)

        # self.extra_repr()
        self.__repr__()

    def extra_repr(self):
        s = ('im_size={im_size}, ht_size={ht_size}')
        # return s.format(**self.__dict__)
        return print(s.format(**self.__dict__))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'im_size=' + str(self.im_size) + ', ht_size=' + str(self.ht_size) \
               + ', vote_mapping=' + str(self.vote_mapping.shape) +  ')'


    def forward(self, input):  
 
        return HT2IMFunction.apply(
            input.contiguous(),
            self.vote_mapping,
            self.im_size, 
            self.ht_size
        )
