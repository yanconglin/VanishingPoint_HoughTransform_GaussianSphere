import scipy.io as sio
import random
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import time
import os
import collections.abc
container_abcs = collections.abc
from itertools import repeat
import scipy.io as sio
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist
from multiprocessing import Pool, sharedctypes #  Process pool
from multiprocessing import Process, Manager
from neurvps.models.sphere.sphere_utils import fibonacci_sphere, sphere_to_catesian, catersian_to_sphere, cos_cdis
from neurvps.models.sphere.sphere_utils import hough_transform, compute_norm

### Initialize seeds: this is important!
np.random.seed(0)
random.seed(0)

######################################################################################################
rows = 128
cols = 128
theta_res = 1
rho_res = 1

# vote_mapping, rho, theta = hough_transform(rows, cols, theta_res, rho_res)
# h = len(rho)
# w = len(theta)

# ht_npz_name= f"vote_index_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
# np.savez(ht_npz_name,
#             vote_mapping=vote_mapping,
#             rho=rho,
#             theta=theta,
#             rows=rows,
#             cols=cols,
#             h=h,
#             w=w,
#             theta_res=theta_res,
#             rho_res=rho_res)

h = 183
w = 180
ht_npz_name= f"../ht/vote_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
ht_npzfile = np.load(ht_npz_name, allow_pickle=True)
vote_mapping_ht = ht_npzfile['vote_mapping']
vote_mapping_ht = vote_mapping_ht.astype(np.float32)
h, w = ht_npzfile["h"],  ht_npzfile["w"]
rho, theta = ht_npzfile["rho"],  ht_npzfile["theta"]
print('vote_index_ht', vote_mapping_ht.shape)

focal_length=2.1875  # wireframe
# focal_length=2.408333333333333 # scannet
# focal_length=1.0 # natural

vote_norm = compute_norm(vote_mapping_ht, rows, cols, h, w, rho, theta, focal_length)
print('vote_norm', vote_norm.shape,(vote_norm.sum(axis=-1)==0.0).sum())

###############  sphere parameterization (parallel computation) ##########################
num_points = 1024
sphere_size = 32768//2

alphas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_points + 1, dtype=np.float32)
alphas_ = alphas_[0:-1]

### make vote_matrix_sphere and vote_index_sphere negative!
vote_sphere = np.ctypeslib.as_ctypes(np.zeros((h*w, num_points*3), dtype=np.float32))
shared_array = sharedctypes.RawArray(vote_sphere._type_, vote_sphere)


def gnomonic_transform(i):
    #####################################################################
    tmp = np.ctypeslib.as_array(shared_array)

    print('i', i, w*h)
    norm_vector = vote_norm[i//w, i%w, :]
    # print('nomr_vector', norm_vector)

    if np.sum(np.abs(norm_vector)) > 0.0:
        if np.abs(norm_vector[1].item())<=1e-16:
            alpha_0 = np.arctan(- norm_vector[2]/(max(norm_vector[0], 1e-16) if norm_vector[0]>=0.0 else min(norm_vector[0], -1e-16)))
            betas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_points + 1, dtype=np.float32)
            betas_ = betas_[0:-1]
            angles_ = np.concatenate((np.ones((num_points, 1))*alpha_0.item(), betas_[:, None]), axis=1)
            xyz_ = sphere_to_catesian(angles_)
        else:
            betas_ = np.arctan(
                (-norm_vector[0] * np.sin(alphas_) - norm_vector[2] * np.cos(alphas_)) / (max(norm_vector[1], 1e-16) if norm_vector[1]>=0.0 else min(norm_vector[1], -1e-16)))
            angles_ = np.concatenate((alphas_[:,None], betas_[:, None]), axis=1)
            xyz_ = sphere_to_catesian(angles_)
        tmp[i,:]=xyz_.flatten()


idxs =np.arange(w*h)
p = Pool()
res = p.map(gnomonic_transform, idxs)
vote_sphere = np.ctypeslib.as_array(shared_array)
vote_sphere = vote_sphere.astype(np.float32).reshape(h, w, num_points, 3)
assert vote_sphere.max()>0.0
print('vote_sphere', vote_sphere.shape)
# #
# # ### save the computed parameterization ##########################
# npz_name = f"vote_matrix_{h:d}_{w:d}_{num_points:d}.npz"
# np.savez(npz_name,
#          vote_matrix=vote_matrix_sphere,
#            num_points=num_points,
#            focal_length=focal_length,
#            vote_norm=vote_norm)
# # #

angles, xyz = fibonacci_sphere(num_bins=sphere_size)
print('angles', angles.shape)
print('xyz', xyz.shape)

def vote2sphere(idx):
    print('idx', idx)
    cur_xyz=vote_sphere[idx//w, idx%w, :, :]
    nonzero_idx =  np.abs(cur_xyz).sum(1)>0.0
    if nonzero_idx.sum()==0: return []
    cur_xyz = cur_xyz[nonzero_idx, :]

    # dist_cos = cos_cdis(cur_xyz, xyz, semi_sphere=True)
    # dist_cos_min, dist_cos_argmin = np.min(dist_cos, axis=1), np.argmin(dist_cos, axis=1)

    dist_cos = cur_xyz @ np.transpose(xyz)
    ### arcos:  y = arcos(x);  (x, y): (1, 0), (0, pi/2), (-1. pi)
    dist_cos_max, dist_cos_argmax = np.max(np.abs(dist_cos), axis=1), np.argmax(np.abs(dist_cos), axis=1)

    mapping = []
    unq, count = np.unique(dist_cos_argmax, return_counts=True)
    # print('unq', unq, count)
    for cur_unq in unq:
        value = dist_cos_max[dist_cos_argmax==cur_unq].max()
        # value *= 180.0/np.pi
        mapping.append([idx, cur_unq, value])
        # vote_mapping.append(np.array([idx//w, idx%w, cur_unq, value]))
        # print('mapping', len(mapping))
    return mapping


idxs =np.arange(w*h)
p = Pool()
total_mappings = p.map(vote2sphere, idxs)
vote_mapping_sphere = [ele for sublist in total_mappings for ent in sublist for ele in ent]
vote_mapping_sphere = np.array(vote_mapping_sphere).reshape(-1, 3)
print('vote_mapping_sphere', vote_mapping_sphere.shape)


npz_name = f"vote_{rows:d}_{cols:d}_{h:d}_{w:d}_{num_points:d}_{sphere_size:d}.npz"

np.savez(npz_name,
         vote_mapping_ht=vote_mapping_ht,
         rho=rho,
         theta=theta,
         rows=rows,
         cols=cols,
         h=h,
         w=w,
         theta_res=theta_res,
         rho_res=rho_res,

         vote_sphere=vote_sphere,
         vote_norm=vote_norm,
           num_points=num_points,
           sphere_size=sphere_size,
           angles=angles,
           xyz=xyz,
           focal_length=focal_length,
           vote_mapping_sphere=vote_mapping_sphere)


# # # https://stackoverflow.com/questions/25411673/python-multiprocessing-appending-list
# def func(inputs):
#     successes = []
#
#     for input in inputs:
#         result = #something with return code
#         if result == 0:
#             successes.append(input)
#     return successes
#
# def main():
#     pool = mp.Pool()
#     total_successes = pool.map(func, myInputs) # Returns a list of lists
#     # Flatten the list of lists
#     total_successes = [ent for sublist in total_successes for ent in sublist]

# # # https://stackoverflow.com/questions/42490368/pythonappending-to-the-same-list-from-different-processes-using-multiprocessing
# if __name__ == "__main__":
#     with Manager() as manager:
#         vote_mapping = manager.list()  # <-- can be shared between processes.
#         processes = []
#         idxs =np.arange(w*h)
#         with Pool() as p:
#             for a, b in p.map(vote2sphere, idx):  # or   imap()
#         res = p.map(vote2sphere, idxs)
#
#         for i in range(4):
#             p = Process(target=vote2sphere, args=(L, i))  # Passing the list
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()
#
#

# # ###### trilinear interpolation ##################
# # vote_pos = vote_matrix.reshape(-1, 3)
# # vote_pos = vote_pos[vote_pos.sum(-1)!=-3.0]
# # print('vote_pos', vote_pos.shape)
# # #
# # # from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, griddata
# # # values = np.ones((len(vote_pos),))
# # # grid_interpolate = griddata(vote_pos, values, xyz, method='linear')
# # # print('grid_interpolate', grid_interpolate.shape)
# # print('xyz', xyz.shape, xyz.max(), xyz.min(), xyz[:,0].min(), xyz[:,1].min(), xyz[:,2].min() )
#
#
#
#
