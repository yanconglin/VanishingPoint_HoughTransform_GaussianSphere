import os.path as osp
import torch
import numpy as np
import numpy.linalg as LA
import math
import scipy.spatial.distance as scipy_spatial_dist
import matplotlib.pyplot as plt


def nms(prob, pos, n_points):
    # pos is positions, prob is probabilities, n_points the number of points to return
    # print('prob, pos, n_points', prob.shape, pos.shape, n_points)
    # result array
    idx = []
    idx.append(prob.argmax().item())

    # keep track of original indices for future reference
    original_idx = torch.arange(pos.size(0))

    for _ in range(1, n_points):
        # print('_', _)
        # mask initial region, matmul instead of dot, because dot only works on same size vectors
        # pos shrinks each time the loop runs. Example sizes of pos:
        # i = 0: 1024
        # i = 1: 512
        # i = 2: 256
        # ... etc.
        dist = pos @ pos[idx].T
        # print('dist', dist.shape)
        dist = torch.max(dist.abs(), dim=1)[0]
        mask = torch.acos(dist) >= math.pi/n_points # angle is orthogonal or larger

        # mask original indices, positions and probabilities
        idx_temp = original_idx[mask]
        prob_temp = prob[mask]

        # new maximum target
        idx.append(idx_temp[prob_temp.argmax()].item())

    return torch.as_tensor(idx)
