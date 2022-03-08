# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as LA
import os
import argparse
import torch
import matplotlib.pyplot as plt
import random
import math
import glob
import skimage.io
import scipy.optimize
import sklearn.metrics
import scipy.sparse
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch

colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
           '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
           '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']


def single_eval_nyu(true_vps, estm_vps, separate_errors=True, normalised_coords=True, missing_vp_penalty=90.):

    ### camera intrinsics
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    S = np.matrix([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
    K = np.matrix([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    SK = S * K
    Kinv = K.I
    SKinv = SK.I

    invmat = SKinv if normalised_coords else Kinv

    true_num_vps = true_vps.shape[0]
    true_vds = (invmat * np.matrix(true_vps).T).T
    for vi in range(true_vds.shape[0]):
        true_vds[vi,:] /= np.maximum(np.linalg.norm(true_vds[vi,:]), 1e-16)

    estm_num_vps = estm_vps.shape[0]
    num_vp_penalty = np.maximum(true_num_vps-estm_num_vps, 0)

    missing_vps = -estm_num_vps+true_num_vps

    estm_vds = (invmat * np.matrix(estm_vps).T).T
    for vi in range(estm_vds.shape[0]):
        estm_vds[vi,:] /= np.maximum(np.linalg.norm(estm_vds[vi,:]), 1e-16)

    cost_matrix = np.arccos(np.abs(np.array(true_vds * estm_vds.T))) * 180. / np.pi
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    loss = cost_matrix[row_ind, col_ind].sum() + num_vp_penalty * missing_vp_penalty


    errors = []
    for ri, ci in zip(row_ind, col_ind):
        errors += [cost_matrix[ri,ci]]
    if missing_vp_penalty > 0:
        errors += [missing_vp_penalty for _ in range(num_vp_penalty)]
    if separate_errors:
        return errors, missing_vps, row_ind, col_ind
    else:
        return loss, missing_vps, row_ind, col_ind

def calc_auc(error_array, cutoff=0.25):

    error_array = error_array.squeeze()
    error_array = np.sort(error_array)
    num_values = error_array.shape[0]

    plot_points = np.zeros((num_values, 2))

    midfraction = 1.

    for i in range(num_values):
        fraction = (i + 1) * 1.0 / num_values
        value = error_array[i]
        plot_points[i, 1] = fraction
        plot_points[i, 0] = value
        if i > 0:
            lastvalue = error_array[i - 1]
            if lastvalue < cutoff < value:
                midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

    if plot_points[-1, 0] < cutoff:
        plot_points = np.vstack([plot_points, np.array([cutoff, 1])])
    else:
        plot_points = np.vstack([plot_points, np.array([cutoff, midfraction])])

    sorting = np.argsort(plot_points[:, 0])
    plot_points = plot_points[sorting, :]

    auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= cutoff, 0],
                              plot_points[plot_points[:, 0] <= cutoff, 1])
    auc = auc / cutoff

    return auc, plot_points

def vps_clustering(vps_prob, xyz, threshold):
    inds = np.flatnonzero(vps_prob >= threshold)
    vps = xyz[inds, :]
    dis = vps @ np.transpose(vps)
    dis = np.clip(dis, a_min=-1., a_max=1.)  ### same=1, opposite=-1, orthogonal=0
    dis = 1.0 - np.abs(dis)  ### same/opposite =0, orthogonal = 1

    dis_sparse = scipy.sparse.csr_matrix(dis)
    clusterer = DBSCAN(eps=0.005, min_samples=9, metric='precomputed').fit(dis_sparse)
    labels = clusterer.labels_
    # print('clusters', type(clusters), clusters.shape, np.unique(clusters))

    if labels.min()<=0: labels += (np.abs(labels.min())+1)  ### the labels from DBSCAN can be negtive (zeros) sometimes

    vps_pd=[]
    for label in np.unique(labels):
        inds_cluster = inds[labels==label]
        vp_max, vp_argmax = np.max(vps_prob[inds_cluster]), np.argmax(vps_prob[inds_cluster])
        vps_pd.append(np.array([inds_cluster[vp_argmax], vp_max]))
        # print('vps_pd', inds_cluster[vp_argmax], vp_max, len(inds_cluster))
    vps_pd = np.vstack(vps_pd)

    arg_prob = np.argsort(vps_pd[:, 1])[::-1]
    vps_pd_sort = vps_pd[arg_prob, 0].astype(int)

    # # # cluster labels for each spherical point
    vps_cluster = np.zeros(vps_prob.shape)
    vps_cluster[inds] = labels

    return xyz[vps_pd_sort], vps_cluster




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NYU-VP dataset visualisation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default="/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/dataset/nyu_vp/processed_data", help='where to load')
    parser.add_argument('--pred_dir', default='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yanconglin/vpd/VPS_code/logs/211103-130207-nyu/results_latest_nyu', help='where to save')
    parser.add_argument('--num_points', type=int, default=32768, help='number of spherical points')
    opt = parser.parse_args()

    xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), 90.0*np.pi/180.0, opt.num_points)

    imagelist = sorted(glob.glob(opt.data_dir + "/*_0.png"))[1224:] # test only
    filelist = sorted(glob.glob(opt.pred_dir + "/000???.npz")) # test only
    print('imagelist, filelist', len(imagelist), len(filelist))

    all_errors=[]
    for idx, (iname, file) in enumerate(zip(imagelist, filelist)):
        print('iname', idx, iname, file)
        image = skimage.io.imread(iname).astype(float)[:, :, :3]/255.0

        ############### load label ################################
        gtfile = np.load(iname.replace(".png", f".npz"), allow_pickle=True)
        vpts_gt = gtfile["vpts"]

       ############### load pd ################################
        predfile = np.load(file, allow_pickle=True)
        vpts_sphere = predfile["vpts_sphere"].astype(np.float32)
        vpts_pred, clusters = vps_clustering(vpts_sphere, xyz, threshold=0.5)
        vpts_pred = vpts_pred[0:len(vpts_gt)]  # topk

        errors, _, row_ind, col_ind = single_eval_nyu(vpts_gt, vpts_pred, missing_vp_penalty=90.)
        all_errors += errors

    np.savez_compressed(os.path.join(opt.pred_dir, 'error.npz'), error = np.hstack(all_errors))
    auc, plot_points = calc_auc(np.array(all_errors), cutoff=10)
    print("AUC: ", auc.shape, auc)
    plt.figure()
    plt.plot(plot_points[:, 0], plot_points[:, 1], 'b-', lw=3, label='AUC: %.3f ' % (auc * 100.))
    axes = plt.gca()
    axes.set_xlim([0, 10])
    axes.set_ylim([0, 1])
    plt.xlabel('error threshold', fontsize=14)
    plt.ylabel('recall', fontsize=14)
    plt.legend()
    plt.show()
