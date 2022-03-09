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
import collections.abc
container_abcs = collections.abc
from itertools import repeat
import scipy.io as sio
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist



def intx(x):
    return(int(x[0]), int(x[1]))

def cos_cdis(x, y, semi_sphere=False):
    # input: x: mxp, y: nxp
    # output: y, mxn
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    ### compute cosine distance
    ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
    dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  # num_points_ x num_points
    # ### map to: same 1, opposite -1, orthorgonal 0, dist = AB/(|A||B|)
    dist_cos *= -1.0
    dist_cos += 1.0

    if semi_sphere is True: dist_cos = np.abs(dist_cos) #  dist = abs|AB/(|A||B|)|
    dist_cos_arc = np.arccos(dist_cos)
    return dist_cos_arc

def catersian_to_sphere(xyz):
    #  input: xyz, nx3, in the gaussian_catersian coordinate
    #  output: angles, (alpha, beta), nx2, in the gaussian sphere coordinate
    #  beta: elevation; alpha: azimuth.
    ### norm = [sin(alpha)cos(beta), sin(beta), cos(alpha)cos(beta)]
    num_points = len(xyz)
    angle = np.zeros((num_points,2))
    angle[:,1] = np.arcsin(xyz[:,1])
    inner = xyz[:,0] / np.cos(angle[:,1])
    inner = np.clip(inner, a_min=-1.0, a_max=1.0)
    # inner = np.minimum(inner, 1)
    # inner = np.maximum(inner, -1)
    angle[:,0] = np.arcsin(inner)
    # print("angle", angle)
    return angle


def sphere_to_catesian(angles):
    #  input: angles (alphas, betas), nx2, in the gasussian sphere coordinate
    #  output: xyz (x,y,z), nx3, in the gaussian_catersian coordinate
    #  beta: elevation; alpha: azimuth.
    ### n = [sin(alpha)cos(beta), sin(beta), cos(alpha)cos(beta)]
    num_points = len(angles)
    xyz = np.zeros((num_points,3))
    xyz[:,0] = np.sin(angles[:,0]) * np.cos(angles[:,1])
    xyz[:,1] = np.sin(angles[:,1])
    xyz[:,2] = np.cos(angles[:,0]) * np.cos(angles[:,1])

    return xyz

# # # pts = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi/2, num_pts=16384)
def gold_spiral_sampling_patch(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    # print('v, v1, v2', v, v1, v2)
    # indices = np.arange(num_pts) + 0.66
    indices = np.arange(num_pts) + 0.5
    phi = np.arccos(1 + (np.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


# # # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075
def fibonacci_sphere(num_bins, gaussian_sphere=True):
    num_pts = 2*num_bins ### double the num_bins, since we only need a semi-sphere.
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    ### standard xyz coordinate.
    # where 0 ≤ φ ≤ π is latitude (phi) coming down from the pole,
    # and 0 ≤ θ ≤ 2π is longitude (theta).
    # facing the screen: x-pointing to the right of the screen, y-pointing at you from the screen, z-pointing upwards
    x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
    # print('xyz', x.shape)
    # semi-sphere: keep only the x>=0, remove x<0
    x_ = x[x >= 0.0]
    y_ = y[x >= 0.0]
    z_ = z[x >= 0.0]

    if not gaussian_sphere:
        phi = phi[x >= 0.0]
        theta = theta[x >= 0.0]
        xyz = np.concatenate((x[:, None], y[:, None], z[:, None]), axis=1)
        angles = np.concatenate((phi[:, None], theta[:, None]), axis=1)
        if angles.shape[0] > num_bins:
            angles = angles[0:num_bins, :]
            xyz = xyz[0:num_bins, :]
        return angles, xyz
    ### transform to azimuth and elevation in the gaussian sphere
    ### the gaussian sphere is defined in "Interpreting Perspective Images, Stephen T.Barnard"
    ### link: https://www.sciencedirect.com/science/article/pii/S0004370283800216
    # (x,y,z) -(z, x, y) - (alpha, beta) - the gaussian sphere
    # x, y, z =sin(alpha)cos(beta), sin(beta), cos(alpha)cos(beta)
    xyz_ = np.concatenate((y_[:, None], z_[:, None], x_[:, None]), axis=1)
    # print('xyz_', xyz.shape)

    angles_ = catersian_to_sphere(xyz_)
    # alphas, betas = angles_[:,0], angles_[:,1]
    if angles_.shape[0]>num_bins:
        angles_ =angles_[0:num_bins, :]
        xyz_ =xyz_[0:num_bins, :]

    # postions of sampled points on the Gaussian sphere
    # in the form of (alpha, beta): azimuth and elevation
    # and in the form of (x,y,z)
    return angles_, xyz_



def compute_norm(vote_mapping_ht, h_img, w_img, h_ht, w_ht, rhos, thetas, focal_length):
    # input:
    # vote_mapping_ht: [N, w_ht]

    # output:
    # vote_norm: [h_ht, w_ht, 3]


    # ### compute the intersecting of the line with image boundary
    vote_norm = np.zeros((h_ht, w_ht, 3), dtype=np.float32)
    vote_image = np.zeros((h_img, w_img), dtype=np.float32)
    ht_bin_non_valid =[]

    for i in range(w_ht):
        for j in range(h_ht):
            print('i, j', i, j)
            ht_inds = vote_mapping_ht[:, 1] == j*w_ht+i
            if not (ht_inds==True).any(): continue  ### remove the ht bins without any votes from pixels
            # img_inds = vote_mapping_ht[ht_inds, 0].astype(int)
            # # print('img_inds', img_inds.max()//w_img, img_inds.max()%w_img)
            # vote_image.fill(0.0)
            # vote_image[img_inds//w_img, img_inds%w_img] = 1.0
            # # if vote_image.sum()<16: continue

            rho = rhos[j]
            theta = thetas[i]
            # print('rho, theta', rho, theta)
            ### rho = xcos theta + y sin theta
            # [0, rho/sin_theta] and [rho/cos_theta, 0/]
            sin_theta, cos_theta = np.sin(theta*np.pi/180.0), np.cos(theta*np.pi/180.0)
            if sin_theta ==0.0: sin_theta+=1e-16
            if cos_theta ==0.0: cos_theta+=1e-16
            sin_theta = np.sign(sin_theta) * max(np.abs(sin_theta), 1e-16)
            cos_theta = np.sign(cos_theta) * max(np.abs(cos_theta), 1e-16)
            # print('sin_theta', sin_theta, cos_theta)

            valid_points = []
            ### HT image coordinate (x-y), where the origin is the image center
            # x [-w_img//2+0.5, w_img//2-0.5]
            # y [-h_img//2+0.5, h_img//2-0.5]
            ###  This is much more memory-efficient by shifting the coordinate ####
            # coords_r, coords_w = np.ones((rows, cols)).nonzero()
            # coords = np.concatenate((coords_r[:, None], coords_w[:, None]), axis=1).astype(np.float32)
            # coords += 0.5
            # coords[:, 0] = -1.0 * (coords[:, 0] - rows // 2)
            # coords[:, 1] = coords[:, 1] - cols // 2

            x_boundary = (rho-(h_img//2-0.5)*sin_theta)/cos_theta # [x, h_img//2]
            if x_boundary<=w_img//2 and x_boundary>=-w_img//2:  valid_points.append([x_boundary, h_img//2-0.5])
            # print('x_boundary, h_img//2', x_boundary, h_img//2-0.5)
            x_boundary = (rho-(-h_img//2+0.5)*sin_theta)/cos_theta # [x, 1-h_img//2]
            if x_boundary<=w_img//2 and x_boundary>=-w_img//2:  valid_points.append([x_boundary, -h_img//2+0.5])
            # print("x_boundary, 1-h_img//2",x_boundary, -h_img//2+0.5)

            y_boundary = (rho-(w_img//2-0.5)*cos_theta)/sin_theta # [w_img//2-1, y]
            if y_boundary<=h_img//2 and y_boundary>=-h_img//2:  valid_points.append([w_img//2-0.5, y_boundary])
            # print('w_img//2-1, y_boundary', w_img//2-0.5, y_boundary)
            y_boundary = (rho-(-w_img//2+0.5)*cos_theta)/sin_theta # [-w_img//2, y]
            if y_boundary<=h_img//2 and y_boundary>=-h_img//2:  valid_points.append([-w_img//2+0.5, y_boundary])
            # print('-w_img//2, y_boundary',-w_img//2+0.5, y_boundary)
            if len(valid_points)<2:
                ht_bin_non_valid.append([j,i])
                continue
            valid_points = np.array(valid_points, dtype=np.float32)

            # print("valid_points", valid_points.shape)
            #
            # if len(valid_points)>=1 and i>0:
            #     ### change to the image rows and cols
            #     image_points = valid_points.copy()
            #     image_points[:, 1] *= -1
            #     image_points[:, 1] += h_img//2
            #     image_points[:, 0] += w_img//2
            #     print('image points', image_points)
            #     print(vote_image.sum())
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)
            #     ax.imshow(vote_image)
            #     # ax.scatter(0, image_points, c ='r')
            #     # ax.scatter(0, x_half, c='g')
            #     # ax.scatter(y_half, w_img-1, c='b')
            #     ax.scatter(image_points[:,0], image_points[:,1], c ='r')
            #     ax.scatter(30, 50)
            #     plt.show()

            if len(valid_points)>2:
                dist01 = np.linalg.norm(valid_points[0,:] - valid_points[1,:])
                dist02 = np.linalg.norm(valid_points[0,:] - valid_points[2,:])
                dist12 = np.linalg.norm(valid_points[1,:] - valid_points[2,:])
                if dist01 == max(dist01, dist02, dist12): valid_points = valid_points[0:2, :]
                elif dist02 == max(dist01, dist02, dist12): valid_points = np.delete(valid_points, obj=1, axis=0)
                else: valid_points = valid_points[1:3, :]

            # print('points', points)
            points = np.concatenate([valid_points, focal_length * np.ones((len(valid_points),1), dtype=np.float32)], axis=1)
            points[:,0] /= w_img//2
            points[:,1] /= h_img//2

            norm_vector = np.cross(points[0,:], points[1,:])
            norm_vector /= max(np.linalg.norm(norm_vector), 1e-16)
            # print('norm_vector', norm_vector.size(), norm_vector)
            vote_norm[j,i]=norm_vector
            # print('norm_vector', points, norm_vector)

            # vote_image = vote_index_ht[:, :, i] == j
            # vote_image = vote_image.astype(np.float32)
            # if vote_image.sum()<16: continue
            # fig, axs = plt.subplots(1,2)
            # axs = axs.ravel()
            # ax = axs[0]
            # ax.plot([x_0, 0], [0, y_0])
            # ax.set_xlim(0, w_img)
            # ax.set_ylim(0, h_img)
            # ax = axs[1]
            # ax.imshow(vote_image)
            # plt.show()

    print('ht_bin_non_valid', len(ht_bin_non_valid))
    return vote_norm.astype(np.float32)



def hough_transform(rows, cols, theta_res, rho_res):

    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))
    theta = theta[0:len(theta) - 1]

    ###  Actually,the offset does not have to be this large.
    D = np.sqrt((rows //2) ** 2 + (cols //2) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, int(nrho))
    print('rho gap', rho)

    w = np.size(theta)
    h = np.size(rho)
    cos_value = np.cos(theta * np.pi / 180.0).astype(np.float32)
    sin_value = np.sin(theta * np.pi / 180.0).astype(np.float32)
    sin_cos = np.concatenate((sin_value[None, :], cos_value[None, :]), axis=0)

    ###  This is much more memory-efficient by shifting the coordinate ####
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

