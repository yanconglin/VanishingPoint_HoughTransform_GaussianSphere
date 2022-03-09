
import sys
import time
import math
import random
import itertools
from collections import defaultdict
from itertools import accumulate
import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as LA
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vpd.utils import plot_image_grid
from vpd.config import C, M
from vpd.models.dgcn import DGCN
from vpd.models.bce_loss import BCE_Loss
from vpd.models.ht.ht_cuda import HT_CUDA
from vpd.models.iht.iht_cuda import IHT_CUDA
from vpd.models.sphere.sphere_cuda import SPHERE_CUDA
from vpd.models.convs import HT_CONV, SPHERE_CONV
from vpd.models.nms import nms


class VanishingNet(nn.Module):
    def __init__(self, backbone, vote_ht_dict, vote_sphere_dict):
        super().__init__()
        self.backbone = backbone
        # ### output from the backbone is a conv layer, now add bn+relu
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.ht = HT_CUDA(vote_mapping_dict=vote_ht_dict)
        self.iht = IHT_CUDA(vote_mapping_dict=vote_ht_dict) # not used in this repo
        self.sphere = SPHERE_CUDA(vote_mapping_dict=vote_sphere_dict)
        self.ht_conv = HT_CONV(inplanes=128, outplanes=128)
        
        self.sphere_conv0 = SPHERE_CONV(inplanes=128, outplanes=64)
        self.sphere_conv1 = SPHERE_CONV(inplanes=128, outplanes=64)
        self.sphere_conv2 = SPHERE_CONV(inplanes=128, outplanes=64)
        
        # # # Harmonic surface network configuration
        self.hsn0 = DGCN(nf=[64, 64, 64, 64, 64], num_nodes=C.io.num_nodes[0], num_neighbors=C.io.num_neighbors[0])
        self.hsn1 = DGCN(nf=[64, 64, 64, 64, 64], num_nodes=C.io.num_nodes[1], num_neighbors=C.io.num_neighbors[1])
        self.hsn2 = DGCN(nf=[64, 64, 64, 64, 64], num_nodes=C.io.num_nodes[2], num_neighbors=C.io.num_neighbors[2])
        self.loss = BCE_Loss()
        
    def forward(self, input_dict):
        x = self.backbone(input_dict["image"])[0]
        x = self.bn(x)
        x = self.relu(x)

        x = self.ht(x)
        x_ht = self.ht_conv(x)
        
        if input_dict['eval'] is False:
            x0 = self.sphere(x_ht, input_dict["ind0"]) # [B, C, S]
            x0 = self.sphere_conv0(x0) # [B, C, S]
            b, c, s = x0.shape
            x0 = x0.transpose(1,2) # [B, S_total, C]
            x0 = self.hsn0(x0, input_dict["edge0"])

            x1 = self.sphere(x_ht, input_dict["ind1"].reshape(b, -1)) # [B, C, S]
            x1 = self.sphere_conv1(x1) # [B, C, S]
            b, c, s = x1.shape
            x1 = x1.transpose(1,2) # [B, S, C]
            x1 = self.hsn1(x1.reshape(-1, C.io.num_nodes[1], c), input_dict["edge1"].reshape(-1, 2, C.io.num_nodes[1]*C.io.num_neighbors[1]))
            
            x2 = self.sphere(x_ht, input_dict["ind2"].reshape(b, -1)) # [B, C, S]
            x2 = self.sphere_conv2(x2) # [B, C, S]
            b, c, s = x2.shape
            x2 = x2.transpose(1,2) # [B, S, C]
            x2 = self.hsn2(x2.reshape(-1, C.io.num_nodes[2], c), input_dict["edge2"].reshape(-1, 2, C.io.num_nodes[2]*C.io.num_neighbors[2]))
            # print('x', x0.shape, x1.shape, x2.shape)

            losses = {}
            loss_pos0, loss_neg0 = self.loss(x0, input_dict['target0'])
            loss_pos1, loss_neg1 = self.loss(x1, input_dict["target1"])
            loss_pos2, loss_neg2 = self.loss(x2, input_dict["target2"])
            # print('loss_pos, loss_neg', loss_pos0.data, loss_neg0.data, loss_pos1.data, loss_neg1.data, loss_pos2.data, loss_neg2.data)

            losses["loss_pos0"] = loss_pos0 * M.lpos
            losses["loss_neg0"] = loss_neg0 * M.lneg
            losses["loss_pos1"] = loss_pos1 * M.lpos
            losses["loss_neg1"] = loss_neg1 * M.lneg
            losses["loss_pos2"] = loss_pos2 * M.lpos
            losses["loss_neg2"] = loss_neg2 * M.lneg

            return {
                "losses": [losses],
                "pred0":  x0.sigmoid().view(b, -1),
                "pred1":  x1.sigmoid().view(b, -1),
                "pred2":  x2.sigmoid().view(b, -1),
            }
        else:
            # print('############################################################')
            ##### mode: batch==1
            x0 = self.sphere(x_ht, input_dict["ind0"]) # [B, C, S]
            x0 = self.sphere_conv0(x0) # [B, C, S]
            b, c, s = x0.shape
            x0 = x0.transpose(1,2) # [B, S_total, C]
            x0 = self.hsn0(x0, input_dict["edge0"])
            
            #####################################################################
            xyz = input_dict["xyz"][0]
            ind0 = input_dict["ind0"][0]
            ind1_scale = input_dict["ind1"][0]
            ind2_scale = input_dict["ind2"][0]
            xyz0 = xyz[ind0]
            
            idx1 = nms(x0.flatten(), xyz0, 3)
            anchor1 = ind0[idx1]
            vpts1 = xyz[anchor1]
            
            ind1 = ind1_scale[anchor1] # [3, num_nodes]
            xyz1 = xyz[ind1.flatten()].view(3, -1, 3)  # [3, num_nodes, 3]
                        
            dis1 = torch.bmm(xyz1, xyz1.transpose(1,2)) # [3, num_nodes, num_nodes]
            _, neighbor1 = dis1.abs().topk(dim=2, k=C.io.num_neighbors[1])
            center1 = torch.arange(C.io.num_nodes[1], device=x.device,dtype=torch.long).repeat_interleave(C.io.num_neighbors[1]).view(1,-1).expand(3, -1)
            edge1 = torch.stack([center1, neighbor1.view(3, -1)], dim=1) # [3, 2, num_nodes*num_neighbors]
            
            x1 = self.sphere(x_ht, ind1[None].reshape(b, -1)) # [B, S]
            x1 = self.sphere_conv1(x1) # [B, C, S]
            b, c, s = x1.shape
            x1 = x1.transpose(1,2) # [B, S, C]
            x1 = self.hsn1(x1.reshape(b*3, s//3, c), edge1)

            #####################################################################
            idx2 = x1.flatten().view(3, s//3).argmax(dim=1)
            anchor2 = ind1.gather(index=idx2[:, None], dim=1)
            vpts2 = xyz[anchor2.flatten()]

            ind2 = ind2_scale[anchor2.flatten()] # [3, num_nodes]
            xyz2 = xyz[ind2.flatten()].view(3, -1, 3)  # [3, num_nodes, 3]
            # print('2, vpts, idx, anchor, ind. xyz, x1', vpts2.shape, idx2, anchor2.shape, ind2.shape, xyz2.shape, x1.shape)

            dis2 = torch.bmm(xyz2, xyz2.transpose(1,2)) # [3, num_nodes, num_nodes]
            _, neighbor2 = dis2.abs().topk(dim=2, k=C.io.num_neighbors[2])
            center2 = torch.arange(C.io.num_nodes[2], device=x.device,dtype=torch.long).repeat_interleave(C.io.num_neighbors[2]).view(1,-1).expand(3, -1)
            edge2 = torch.stack([center2, neighbor2.view(3, -1)], dim=1) # [3, 2, num_nodes*num_neighbors]

            x2 = self.sphere(x_ht, ind2[None].reshape(b, -1)) # [B, S]
            x2 = self.sphere_conv2(x2) # [B, C, S]
            b, c, s = x2.shape
            x2 = x2.transpose(1,2) # [B, S, C]
            x2 = self.hsn2(x2.reshape(3, s//3, c), edge2)

            idx3 = x2.flatten().view(3, s//3).argmax(dim=1)
            anchor3 = ind2.gather(index=idx3[:, None], dim=1)
            vpts3 = xyz[anchor3.flatten()]
            # print('3, vpts, idx, anchor, x2', vpts3.shape, idx3, anchor3.shape, x2.shape)

            return {
                "pred0":  x0.sigmoid().view(-1),
                "pred1":  x1.sigmoid().view(3, -1),
                "pred2":  x2.sigmoid().view(3, -1),

                "idx1": idx1,
                "anchor1": anchor1,
                "ind1": ind1,

                "idx2": idx2,
                "anchor2": anchor2,
                "ind2": ind2,

                "idx3": idx3,
                "anchor3": anchor3,

                "vpts1":  vpts1,
                "vpts2":  vpts2,
                "vpts3":  vpts3,
            }

