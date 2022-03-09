import sys
import math
import random
import itertools
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as LA
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def make_conv2d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    if bias is False: layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def make_conv1d_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
    layers = []
    layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    if bias is False: layers += [nn.BatchNorm1d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)



def init_ht_convs(layers, laplacian_init=False):
    if not laplacian_init:
        nn.init.kaiming_normal_(layers.weight, mode='fan_out', nonlinearity='relu')
        if layers.bias is not None: nn.init.zeros_(layers.bias.data)
    else:
        print('Laplacian intialization')
        # # # 1-D filters (k, 1), defalult (9x1)
        out_channels, in_channels, h, w = layers.weight.data.shape
        assert w == 1
        z = []
        for k in range(0, out_channels):
            for kk in range(0, in_channels):
                x = np.zeros(shape=((h)))
                x[(h - 1) // 2] = 1
                sigma = np.random.uniform(low=1.0, high=3.0, size=(1))
                y = ndimage.filters.gaussian_filter(x, sigma=sigma, order=2)
                y = -y / np.sum(np.abs(y))
                z.append(y)
        z = np.stack(z).reshape(out_channels, in_channels, h)
        layers.weight.data.copy_(torch.from_numpy(z).unsqueeze(-1))
        if layers.bias is not None: layers.bias.data.fill_(0.0)


class HTCONVBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(9,1), stride=1, padding=(4,0), dilation=1, groups=1, bias=True)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(9,1), stride=1, padding=(4,0), dilation=1, groups=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resample = nn.Identity()
        if in_planes != planes:
            self.resample = nn.Conv2d(in_planes, planes, 1)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        # out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out += self.resample(residual)

        return self.relu(out)



class HT_CONV(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(HT_CONV, self).__init__()

        self.conv1 = nn.Sequential(*make_conv2d_block(inplanes, inplanes, kernel_size=(9,1), padding=(4,0), bias=True, groups=inplanes))
        self.block1 = HTCONVBlock(inplanes, inplanes)
        self.block2 = HTCONVBlock(inplanes, outplanes)

        self.relu = nn.ReLU(inplace=True)

        init_ht_convs(self.conv1[0], laplacian_init=True)
        init_ht_convs(self.block1.conv1, laplacian_init=False)
        init_ht_convs(self.block1.conv2, laplacian_init=False)
        init_ht_convs(self.block2.conv1, laplacian_init=False)
        init_ht_convs(self.block2.conv2, laplacian_init=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class SPHERE_CONV(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SPHERE_CONV, self).__init__()

        self.conv1 = nn.Sequential(*make_conv1d_block(inplanes, outplanes, kernel_size=1, bias=False))

        init_ht_convs(self.conv1[0], laplacian_init=False)

    def forward(self, x):
        x = self.conv1(x)
        return x

