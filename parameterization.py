import os
import argparse
import time
import random
import numpy as np
import torch
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import collections.abc
container_abcs = collections.abc
from itertools import repeat
import scipy.io as sio
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist
from scipy.interpolate import griddata

def cos_cdis(x, y, semi_sphere=False):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
    dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine') 
    dist_cos *= -1.0
    dist_cos += 1.0

    if semi_sphere is True: dist_cos = np.abs(dist_cos) #  dist = abs|AB/(|A||B|)|
    dist_cos_arc = np.arccos(dist_cos)
    return dist_cos_arc

def catersian_to_sphere(xyz):
    #  beta: elevation; alpha: azimuth.
    ### norm = [sin(alpha)cos(beta), sin(beta), cos(alpha)cos(beta)]
    num_points = len(xyz)
    angle = np.zeros((num_points,2))
    angle[:,1] = np.arcsin(xyz[:,1])
    inner = xyz[:,0] / np.cos(angle[:,1])
    inner = np.clip(inner, a_min=-1.0, a_max=1.0)
    angle[:,0] = np.arcsin(inner)
    return angle


def sphere_to_catesian(angles):
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


def to_pixel(vpts, focal_length=1.0, h=480, w=640):
    x = vpts[:,0] / vpts[:, 2] * focal_length * max(h, w)/2.0 + w//2
    y = -vpts[:,1] / vpts[:, 2] * focal_length * max(h, w)/2.0 + h//2
    return y, x


def hough_transform(rows, cols, theta_res, rho_res):

    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))+0.5
    theta = theta[0:len(theta) - 1]
    D = np.sqrt((rows//2 +0.5) ** 2 + (cols//2+0.5) ** 2)
    rho = np.arange(-D, D+rho_res, rho_res)
    # print('rho', rho)

    w = np.size(theta)
    h = np.size(rho)
    cos_value = np.cos(theta * np.pi / 180.0)
    sin_value = np.sin(theta * np.pi / 180.0)
    sin_cos = np.concatenate((sin_value[None, :], cos_value[None, :]), axis=0)

    # memory-efficient to centralize the coordinate
    coords_r, coords_w = np.ones((rows, cols)).nonzero()
    coords = np.concatenate((coords_r[:,None], coords_w[:,None]), axis=1).astype(np.float32)
    coords += 0.5

    coords[:,0] = -1.0*(coords[:,0]-rows//2)
    coords[:,1] = coords[:,1] - cols//2

    vote_map = (coords @ sin_cos).astype(np.float32)

    mapping = []
    for i in range(rows*cols):
        print('ith pixel', i//cols, i%cols)
        for j in range(w):
            rhoVal = vote_map[i, j]
            assert rhoVal<=rho.max() or rhoVal>=rho.min()
            # rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
            dis = (rho - rhoVal)
            argsort = np.argsort(np.abs(dis))
            mapping.append(np.array([ i, argsort[0]*w+j, dis[argsort[0]]]))

    return np.vstack(mapping).astype(np.float32), rho.astype(np.float32), theta.astype(np.float32)


def compute_normal(mapping_ht, h_img, w_img, h_ht, w_ht, rhos, thetas, focal_length):
    ht_normal = np.zeros((h_ht, w_ht, 3))
    ht_bin_invalid =[]

    for i in range(w_ht):
        for j in range(h_ht):
            print('i, j', i, j)
            ht_inds = mapping_ht[:, 1] == j*w_ht+i
            if not (ht_inds==True).any(): continue  ### remove ht bins without any votes from pixels

            rho = rhos[j]
            theta = thetas[i]
            # print('rho, theta', rho, theta)
            # line parameterization: rho = xcos theta + y sin theta
            sin_theta, cos_theta = np.sin(theta*np.pi/180.0), np.cos(theta*np.pi/180.0)
            # warning: check edge cases:
            if sin_theta ==0.0: sin_theta+=1e-16
            if cos_theta ==0.0: cos_theta+=1e-16
            sin_theta = np.sign(sin_theta) * max(np.abs(sin_theta), 1e-16)
            cos_theta = np.sign(cos_theta) * max(np.abs(cos_theta), 1e-16)

            valid_points = []
            # This calculation is tedious, but it is easier to debug/visualize
            ### coordinate (x-y), where the origin is the image center
            # x [-w_img//2+0.5, w_img//2-0.5]
            # y [-h_img//2+0.5, h_img//2-0.5]
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
                ht_bin_invalid.append([j,i])
                continue
            valid_points = np.array(valid_points)

            # Again: tedious calculation, not necessary at all
            # here I chose the longest line segment, you can simply select any two.
            if len(valid_points)>2:
                dist01 = np.linalg.norm(valid_points[0,:] - valid_points[1,:])
                dist02 = np.linalg.norm(valid_points[0,:] - valid_points[2,:])
                dist12 = np.linalg.norm(valid_points[1,:] - valid_points[2,:])
                if dist01 == max(dist01, dist02, dist12): valid_points = valid_points[0:2, :]
                elif dist02 == max(dist01, dist02, dist12): valid_points = np.delete(valid_points, obj=1, axis=0)
                else: valid_points = valid_points[1:3, :]

            points = np.concatenate([valid_points, focal_length * np.ones((len(valid_points),1))], axis=1)
            points[:,0] /= max(h_img, w_img)//2
            points[:,1] /= max(h_img, w_img)//2
            
            normal_vector = np.cross(points[0,:], points[1,:])
            normal_vector /= max(np.linalg.norm(normal_vector), 1e-16)
            if normal_vector[2]<0.0: normal_vector *= -1
            ht_normal[j,i]=normal_vector

            # vote_image = vote_index_ht[:, :, i] == j
            # vote_image = vote_image.astype(np.float32)
            # fig, axs = plt.subplots(1,2)
            # axs = axs.ravel()
            # ax = axs[0]
            # ax.plot([x_0, 0], [0, y_0])
            # ax.set_xlim(0, w_img)
            # ax.set_ylim(0, h_img)
            # ax = axs[1]
            # ax.imshow(vote_image)
            # plt.show()

    print('warning: invalid ht bins', len(ht_bin_invalid)) # ignore, not a problem
    return ht_normal.astype(np.float32)


def compute_sphere(ht_normal, num_samples):

    alphas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_samples + 1, dtype=np.float64)
    alphas_ = alphas_[0:-1]

    betas_ = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=num_samples + 1, dtype=np.float64)
    betas_ = betas_[0:-1]

    all_hw_xyz = []

    h, w, _= ht_normal.shape
    for i in range(0, h*w):
        cur_xyz = []
        norm_vector = ht_normal[i//w, i%w, :]
        if not np.sum(np.abs(norm_vector)) > 0.0: continue  # invalid ht bins
        print('i', i, w*h, norm_vector)
        if np.abs(norm_vector[1].item()) <= 1e-16:
            # warning: edge cases
            alpha_0 = np.arctan(- norm_vector[2] / (max(norm_vector[0], 1e-16) if norm_vector[0] >= 0.0 else min(norm_vector[0], -1e-16)))
            angles_alphas_0_betas = np.concatenate((np.ones((num_samples, 1)) * alpha_0.item(), betas_[:, None]), axis=1)
            xyz_alphas_0_betas = sphere_to_catesian(angles_alphas_0_betas)
            cur_xyz.append(xyz_alphas_0_betas)
        else:
            # infer betas from alphas
            betas_f = np.arctan((-norm_vector[0] * np.sin(alphas_) - norm_vector[2] * np.cos(alphas_)) /
                                (max(norm_vector[1], 1e-16) if norm_vector[1] >= 0.0 else min(norm_vector[1], -1e-16)))
            angles_alphas_betas_f = np.concatenate((alphas_[:, None], betas_f[:, None]), axis=1)
            xyz_alphas_betas_f = sphere_to_catesian(angles_alphas_betas_f)
            cur_xyz.append(xyz_alphas_betas_f)

            # infer alphas from betas
            # solution 1
            # x sin(alpha) + z cos(alpha) = -y tan(beta)
            # solved by using sin(alpha) cos(phi) + cos(alpha) sin(phi) = sin(alpha+phi)
            # sin(alpha+phi) = -y tan(beta), tan(phi) = z/x
            # solution 2:
            # x sin(alpha) + z cos(alpha) = -y tan(beta)
            # cos(alpha-phi) = -y tan(beta)/sqrt(x**2+z**2), tan(phi) = x/z;
            norm_xz = np.sqrt(norm_vector[0] ** 2 + norm_vector[2] ** 2)
            norm_xz = max(norm_xz, 1e-16)
            t = -norm_vector[1] * np.tan(betas_) / norm_xz
            t_inds = np.logical_and(t >= -1, t <= 1)  # check cos in [-1,1]
            if t_inds.sum() == 0.0: continue  # skip
            t = t[t_inds]
            betas = betas_[t_inds]

            # calculate alpha and phi
            # cos(alpha-phi) = -y tan(beta)/sqrt(x**2+z**2)
            alphas_phi = np.arccos(t)  # alpha-phi: [0, pi]
            # tan(phi) = x/z;
            x_z = norm_vector[0] / max(norm_vector[2], 1e-16) if norm_vector[2] >= 0 else norm_vector[0] / min(norm_vector[2], -1e-16)
            phi = np.arctan(x_z)  # [-pi/2, pi/2]

            # Symmetry/periodicity of cosine
            alphas_f1 = alphas_phi + phi
            alphas_f2 = -1.0*alphas_phi + phi
            # (1) choose alphas_f2 if alphas_f1>=np.pi/2 OR <-np.pi/2
            inds1 = np.logical_and(alphas_f1>=-np.pi/2, alphas_f1<np.pi/2)
            alphas_f = np.where(inds1, alphas_f1, alphas_f2)
            # (2) check -np.pi/2=<alphas_f<np.pi/2
            inds2 = np.logical_and(alphas_f>=-np.pi/2, alphas_f<np.pi/2)
            alphas_f = alphas_f[inds2]
            # (3) update betas
            betas = betas[inds2]

            angles_alphas_f_betas = np.concatenate((alphas_f[:, None], betas[:, None]), axis=1)
            xyz_alphas_f_betas = sphere_to_catesian(angles_alphas_f_betas)

            # hemi-sphere on the side of z >=0
            z_inds = xyz_alphas_f_betas[:,2]>=0.0
            xyz_alphas_f_betas = xyz_alphas_f_betas[z_inds,:]
            cur_xyz.append(xyz_alphas_f_betas)

        cur_xyz = np.vstack(cur_xyz)

        # visualize: to debug
        # fig = plt.figure()
        # ax = fig.add_subplot(131, projection='3d',)
        # # ax.scatter(cur_xyz[:, 0], cur_xyz[:, 1], cur_xyz[:, 2], c='r')
        # if np.abs(norm_vector[1].item()) <= 1e-16:
        #     ax.scatter(xyz_alphas_0_betas[:, 0], xyz_alphas_0_betas[:, 1], xyz_alphas_0_betas[:, 2], c='m')
        #
        # else:
        #     ax.scatter(xyz_alphas_f_betas[:, 0], xyz_alphas_f_betas[:, 1], xyz_alphas_f_betas[:, 2], c='b')
        #     ax.scatter(xyz_alphas_betas_f[:, 0], xyz_alphas_betas_f[:, 1], xyz_alphas_betas_f[:, 2], c='g')
        # ax = fig.add_subplot(132, projection='3d',)
        # if np.abs(norm_vector[1].item()) <= 1e-16:
        #     ax.scatter(xyz_alphas_0_betas[:, 0], xyz_alphas_0_betas[:, 1], xyz_alphas_0_betas[:, 2], c='m')
        #
        # else:
        #     ax.scatter(cur_xyz[:, 0], cur_xyz[:, 1], cur_xyz[:, 2], c='r')
        #     ax.scatter(xyz_alphas_f_betas[:, 0], xyz_alphas_f_betas[:, 1], xyz_alphas_f_betas[:, 2], c='b')
        # ax = fig.add_subplot(133, projection='3d',)
        # if np.abs(norm_vector[1].item()) <= 1e-16:
        #     ax.scatter(xyz_alphas_0_betas[:, 0], xyz_alphas_0_betas[:, 1], xyz_alphas_0_betas[:, 2], c='m')
        #
        # else:
        #     ax.scatter(cur_xyz[:, 0], cur_xyz[:, 1], cur_xyz[:, 2], c='r')
        #     ax.scatter(xyz_alphas_betas_f[:, 0], xyz_alphas_betas_f[:, 1], xyz_alphas_betas_f[:, 2], c='g')
        # plt.suptitle(f'{i:d}')
        # plt.show()

        cur_hw = np.zeros((len(cur_xyz),1), dtype=np.float64)
        cur_hw.fill(float(i))
        cur_hw_xyz = np.concatenate([cur_hw, cur_xyz], axis=-1)
        # print('i, cur_hw_xyz', i, w*h, cur_hw_xyz.shape)
        all_hw_xyz.append(cur_hw_xyz)
    all_hw_xyz = np.vstack(all_hw_xyz)
    print('all_hw_xyz', all_hw_xyz.shape)
    return all_hw_xyz.astype(np.float32) # [M, 4]


### find nearest neighbors on the sphere for each hw_xyz in all_hw_xyz
def find_nn_neighbors_sphere(all_hw_xyz, xyz):

    if use_gpu is True or device_name=='cuda':
        torch.cuda.empty_cache()
        all_hw_xyz = torch.from_numpy(all_hw_xyz).float().cuda()
        xyz = torch.from_numpy(xyz).float().cuda()
    else:
        all_hw_xyz = torch.from_numpy(all_hw_xyz).float()
        xyz = torch.from_numpy(xyz).float()

    all_hw = all_hw_xyz[:,0] # M
    all_xyz = all_hw_xyz[:,1:] # Mx3
    all_xyz /= torch.norm(all_xyz, dim=1, keepdim=True)
    mapping = []

    # loop over each ht bin, find its nn neighbors on the sampled sphere
    max_hw = all_hw.max().long().item()
    max_length = 0    
    total_length = 0
    for hw_ind in range(0, max_hw):
        hw_inds = all_hw==hw_ind
        if hw_inds.sum()==0.0: continue
        clip_xyz = all_xyz[hw_inds,:]

        dist_cos = xyz @ clip_xyz.t()  ### N X len(clip_xyz)
        dist_cos = dist_cos.abs()
        max_values, max_inds = torch.max(dist_cos, dim=0)  # len(clip_xyz)

        ### it is possible that clip_xyz share the same nearest neighbors
        unique_inds = torch.unique(max_inds)
        total_length += len(unique_inds)
        if len(unique_inds) > max_length: max_length = len(unique_inds)
        clip_mapping = clip_xyz.new_zeros(len(unique_inds), 3).float()
        clip_mapping[:, 0] = hw_ind
        clip_mapping[:, 1] = unique_inds
        for ind, unique_ind in enumerate(unique_inds):
            max_max_value = max_values[max_inds==unique_ind].max()
            # clip_mapping[ind, 1] = unique_ind
            clip_mapping[ind, 2] = max_max_value
            
        # print('clip_mapping/hw_inds/total_length', len(clip_mapping), hw_inds.sum().item(), total_length)
        mapping.append(clip_mapping.cpu().numpy())  # len(unique_inds)x3
        del clip_mapping

        # memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
        # print('max_memory_allocated', memory)
            
    mapping = np.concatenate(mapping, axis=0) # [M, 3] 
    print('mapping, max_length, total_length', mapping.shape, max_length, total_length)
    return mapping


### re-arrange for efficient sampling
def rearrange(sphere_neighbors, fibonacci_xyz):

    max_len = 0
    for i, xyz in enumerate(fibonacci_xyz):
        # print('i', i)
        inds = sphere_neighbors[:, 1]==i
        max_len = max(max_len, inds.sum())
    print('max_len', max_len)
    
    mapping = np.zeros((len(fibonacci_xyz), max_len, 2))
    mapping.fill(-1.0)
    mapping_non = []
    for i, xyz in enumerate(fibonacci_xyz):
        inds = sphere_neighbors[:, 1] == i
        if inds.sum()==0:
            mapping_non.append(i) # spherical points w/o votes
            continue
        ht = sphere_neighbors[inds, 0]
        weights = sphere_neighbors[inds, 2]
        mapping[i, 0:inds.sum(), 0] = ht
        mapping[i, 0:inds.sum(), 1] = weights

    print('mapping_non', mapping_non)
    print('mapping', mapping.shape)
    return votes

###########################################################
def main(opt):
    ### Initialize seeds: this is important!
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    global use_gpu, device_name
    if torch.cuda.is_available():
        use_gpu=True
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        use_gpu=False
        device_name = "cpu"
        print("CUDA is not available")
    assert use_gpu is True   # use GPUs to speedup calculation: ~ 4 hours

    save_dir = opt.save_dir
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    ### image: [rows, cols], HT: [height, width], Sphere [num_points]
    rows=opt.rows
    cols=opt.cols
    theta_res = opt.theta_res
    rho_res = opt.rho_res
    num_samples = opt.num_samples
    num_points = opt.num_points
    focal_length = opt.focal_length
    fibonacci_xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi/2, num_pts=num_points)

    # ############ HT #####################################
    mapping_ht, rho, theta = hough_transform(rows, cols, theta_res, rho_res)
    print('mapping_ht', mapping_ht.shape)
    h = len(rho)
    w = len(theta)
    ht_npz_name = f"ht_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
    np.savez(os.path.join(save_dir, ht_npz_name),
             ht_mapping=mapping_ht,
             rho=rho,
             theta=theta,
             rows=rows,
             cols=cols,
             h=h,
             w=w,
             theta_res=theta_res,
             rho_res=rho_res)


    ################ normal #####################################
    ht_normal = compute_normal(mapping_ht, rows, cols, h, w, rho, theta, focal_length)
    print('ht_normal', ht_normal.shape, (ht_normal.sum(axis=-1) == 0.0).sum())

    ################ SPHERE #####################################
    all_hw_xyz = compute_sphere(ht_normal, num_samples=num_samples)
    print('all_hw_xyz', all_hw_xyz.shape)

    ############## SPHERE FIND NN NEIGHBORS #####################################
    sphere_neighbors = find_nn_neighbors_sphere(all_hw_xyz, fibonacci_xyz)
    print('sphere_neighbors', sphere_neighbors.shape)

    sphere_neighbors_npz_name = f"sphere_neighbors_{h:d}_{w:d}_{num_points:d}.npz"
    np.savez(os.path.join(save_dir, sphere_neighbors_npz_name),
                h=h, w=w,
                num_points=num_points,
                num_samples=num_samples,
                xyz=fibonacci_xyz,
                focal_length=focal_length,
                sphere_neighbors_weight=sphere_neighbors)
                
                
    ############ rearrange fucniton ##############################################
    # # sphere_neighbors_npz_name = f"sphere_neighbors_{h:d}_{w:d}_{num_points:d}_nn.npz"
    # # sphere_neighbors_npzfile = np.load(os.path.join(dir_name, sphere_neighbors_npz_name), allow_pickle=True)
    # # sphere_neighbors = sphere_neighbors_npzfile['sphere_neighbors']
    # # print('sphere_neighbors', sphere_neighbors.shape, fibonacci_xyz.shape)

    # sphere_neighbors_re = rearrange(sphere_neighbors, fibonacci_xyz)
    # rearrange_npz_name = f"sphere_neighbors_{h:d}_{w:d}_{num_points:d}_rearrange.npz"
    # np.savez(os.path.join(dir_name, rearrange_npz_name),
    #          h=h, w=w,
    #          num_points=num_points,
    #          num_samples=num_samples,
    #          xyz=fibonacci_xyz,
    #          focal_length=focal_length,
    #          sphere_neighbors=sphere_neighbors_re)
    #          
    #          

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', default='/parameterization/', help='path to save parameterizations')
    parser.add_argument('--focal_length', type=float, default=1.0, help='focal length, set to 1.0 if unknown')
    parser.add_argument('--rows', type=int, default=256, help='rows - image height')
    parser.add_argument('--cols', type=int, default=256, help='cols - image width')
    parser.add_argument('--theta', type=float, default=1.0, help='theta - hyperparameter for HT')
    parser.add_argument('--rho', type=float, default=1.0, help='rho - hyperparameter for HT')
    parser.add_argument('--num_samples', type=int, default=180, help='num_samples - number of angles for Gaussian sphere')
    parser.add_argument('--num_points', type=int, default=32768, help='num_points - number of sampled spherical points')
    opt = parser.parse_args()
    print('opt', opt)
    main(opt)
