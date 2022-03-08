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


def hough_transform(rows, cols, theta_res, rho_res):

    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))
    theta = theta[0:len(theta) - 1]

    ###  Actually,the offset does not have to be this large.
    D = np.sqrt((rows //2) ** 2 + (cols //2) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, int(nrho))
    print('rho', rho)

    w = np.size(theta)
    h = np.size(rho)
    cos_value = np.cos(theta * np.pi / 180.0).astype(np.float32)
    sin_value = np.sin(theta * np.pi / 180.0).astype(np.float32)
    sin_cos = np.concatenate((sin_value[None, :], cos_value[None, :]), axis=0)

    ###  This is much more memory-efficient to centralize the coordinate ####
    coords_r, coords_w = np.ones((rows, cols)).nonzero()
    coords = np.concatenate((coords_r[:,None], coords_w[:,None]), axis=1).astype(np.float32)
    coords += 0.5

    coords[:,0] = -1.0*(coords[:,0]-rows//2)
    coords[:,1] = coords[:,1] - cols//2

    vote_map = (coords @ sin_cos).astype(np.float32)
    print('rows, cols, h, w', rows, cols, h, w)

    mapping = []
    for i in range(rows*cols):
        print('ith pixel', i//cols, i%cols)
        for j in range(w):
            rhoVal = vote_map[i, j]
            # print('rhoVal', rhoVal, rho.max(), rho.min())
            assert rhoVal<=rho.max() or rhoVal>=rho.min()
            # rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]

            dis = np.abs(rho - rhoVal)
            argsort = np.argsort(dis)
            # mapping.append(np.array([ i//cols, i%cols, j, argsort[0], (1.0-dis[argsort[0]])/rho_res ]))
            # mapping.append(np.array([ i//cols, i%cols, j, argsort[1], (1.0-dis[argsort[1]])/rho_res ]))
            mapping.append(np.array([ i, argsort[0]*w+j, 1.0-dis[argsort[0]]/rho_res ]))
            if 1.0-dis[argsort[1]]/rho_res>0.0: mapping.append(np.array([ i, argsort[1]*w+j, 1.0-dis[argsort[1]]/rho_res ]))

    return np.vstack(mapping).astype(np.float32), rho.astype(np.float32), theta.astype(np.float32)



if __name__ == "__main__":
    
    rows = 48
    cols = 48
    theta_res = 3
    rho_res = 1
    
    vote_mapping, rho, theta = hough_transform(rows, cols, theta_res, rho_res)
    h = len(rho)
    w = len(theta)

    ht_npz_name= f"vote_index_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
    np.savez(ht_npz_name,
                vote_mapping=vote_mapping,
                rho=rho,
                theta=theta,
                rows=rows,
                cols=cols,
                h=h,
                w=w,
                theta_res=theta_res,
                rho_res=rho_res)