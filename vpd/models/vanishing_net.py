import sys
import math
import random
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as LA
import torch.nn.functional as F
from vpd.config import C, M
from vpd.models.dgcn import DGCN
from vpd.models.bce_loss import BCE_Loss
from vpd.models.ht.ht_cuda import HT_CUDA
from vpd.models.iht.iht_cuda import IHT_CUDA
from vpd.models.sphere.sphere_cuda import SPHERE_CUDA
from vpd.models.convs import HT_CONV, SPHERE_CONV

class VanishingNet(nn.Module):
    def __init__(self, backbone, vote_ht_dict, vote_sphere_dict):
        super().__init__()
        self.backbone = backbone
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.ht = HT_CUDA(vote_mapping_dict=vote_ht_dict)
        self.iht = IHT_CUDA(vote_mapping_dict=vote_ht_dict)
        self.sphere = SPHERE_CUDA(vote_mapping_dict=vote_sphere_dict)

        self.ht_conv = HT_CONV(inplanes=128, outplanes=128)
        self.sphere_conv = SPHERE_CONV(inplanes=128, outplanes=M.num_channels)

        self.hsn = DGCN(nf=[M.num_channels, M.num_channels, M.num_channels, M.num_channels, M.num_channels], num_nodes=C.io.num_nodes, num_neighbors=C.io.num_neighbors)
        self.loss = BCE_Loss()

    def forward(self, input_dict):
        image = input_dict["image"]
        x = self.backbone(image)[0]
        x = self.bn(x)
        x = self.relu(x)

        x = self.ht(x)
        x = self.ht_conv(x)
        x = self.sphere(x)
        x = self.sphere_conv(x)
        x = self.hsn(x)

        loss={}
        loss_pos, loss_neg = self.loss(x, input_dict["target"])
        # print('loss_pos, loss_neg', loss_pos.shape, loss_neg.shape)
        loss["loss_pos"] = loss_pos * M.lpos
        loss["loss_neg"] = loss_neg * M.lneg
        return {
            "losses": [loss],
            "prediction":  x.sigmoid().reshape(-1, C.io.num_nodes),
        }
