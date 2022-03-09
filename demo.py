#!/usr/bin/env python3
"""Compute vanishing points from images.
Usage:
    demo.py [options] <yaml-config> <checkpoint> <image>
    demo.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <image>                       Path to an image

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --dump <output-dir>           Optionally, save the vanishing points to npz format.
"""

import os
import sys
import math
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess
import time
import torch
import matplotlib as mpl
import skimage.io
import numpy as np
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from tqdm import tqdm
from docopt import docopt
import scipy.io as sio
import scipy.optimize
import sklearn.metrics
import scipy.sparse
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import vpd
import vpd.models.vanishing_net as vn
from vpd.config import C, M
from vpd.datasets import ScanNetDataset, WireframeDataset, YUDDataset, NYUDataset
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch, catersian_to_sphere

def topk_orthogonal_vps(scores, xyz, num_vps=3):

    index = np.argsort(-scores)
    vps_idx = [index[0]]
    for i in index[1:]:
        if len(vps_idx) == num_vps:
            break
        # cos_distance function: input: x: mxp, y: nxp; output: y, mxn
        ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
        dist_cos = scipy_spatial_dist.cdist(xyz[vps_idx], xyz[i][None, :], 'cosine')
        ### same 1, opposite -1, orthorgonal 0
        dist_cos = np.abs(-1.0*dist_cos+1.0)

        dist_cos_arc = np.min(np.arccos(dist_cos))
        if dist_cos_arc >= np.pi/num_vps:
            vps_idx.append(i)
        else:
            continue

    vps_pd = xyz[vps_idx]
    return vps_pd, vps_idx

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

    return xyz[vps_pd_sort], vps_cluster.astype(int)



def to_pixel(vpts, focal_length=1.0, h=480, w=640):
    x = vpts[:,0] / vpts[:, 2] * focal_length * max(h, w)/2.0 + w//2
    y = -vpts[:,1] / vpts[:, 2] * focal_length * max(h, w)/2.0 + h//2
    return y, x

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    C.model.im2col_step = 32  # override im2col_step for evaluation
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # # # save plots for visualization	
    # os.environ['QT_QPA_PLATFORM']='offscreen'
    device_name = "cpu"
    num_gpus = args["--devices"].count(",") + 1
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        for k in range(0, torch.cuda.device_count()):
            print('kth, device name', k, torch.cuda.get_device_name(k))
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    npzfile = np.load(C.io.ht_mapping, allow_pickle=True)
    ht_mapping = npzfile['ht_mapping']
    ht_mapping[:,2] = npzfile['rho_res'].item() - np.abs(ht_mapping[:,2])
    ht_mapping[:,2] /= npzfile['rho_res'].item()
    vote_ht_dict={}
    vote_ht_dict["vote_mapping"]= torch.tensor(ht_mapping, requires_grad=False).float().contiguous()
    vote_ht_dict["im_size"]= (npzfile['rows'], npzfile['cols'])
    vote_ht_dict["ht_size"]= (npzfile['h'], npzfile['w'])
    print('vote_ht_dict  memory MB', vote_ht_dict["vote_mapping"].size(),
          vote_ht_dict["vote_mapping"].element_size() * vote_ht_dict["vote_mapping"].nelement() / (1024 * 1024))

    npzfile = np.load(C.io.sphere_mapping, allow_pickle=True)
    sphere_neighbors = npzfile['sphere_neighbors']
    vote_sphere_dict={}
    vote_sphere_dict["vote_mapping"]=torch.tensor(sphere_neighbors, requires_grad=False).float().contiguous()
    vote_sphere_dict["ht_size"]=(npzfile['h'], npzfile['w'])
    vote_sphere_dict["sphere_size"]=npzfile['num_points']
    print('vote_sphere_dict  memory MB', vote_sphere_dict["sphere_size"], vote_sphere_dict["vote_mapping"].size(),
          vote_sphere_dict["vote_mapping"].element_size() * vote_sphere_dict["vote_mapping"].nelement() / (1024 * 1024))


    # 2. model
    if M.backbone == "stacked_hourglass":
        backbone = vpd.models.hg(
            planes=128, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
        )
    else:
        raise NotImplementedError

    model = vpd.models.VanishingNet(backbone, vote_ht_dict, vote_sphere_dict)
    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )

    if args["<checkpoint>"]:
        print('args["<checkpoint>"]', args["<checkpoint>"])
        checkpoint = torch.load(args["<checkpoint>"], map_location=lambda storage, loc: storage)
        print('checkpoint', checkpoint["iteration"], checkpoint["epoch"])
        # print('checkpoint', checkpoint["iteration"])
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # print('model', model)

    ##### number of parameters in a model
    total_params = sum(p.numel() for p in model.parameters())
    ##### number of trainable parameters in a model
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', total_params)
    print('num of trainable parameters', train_params)

    xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.num_nodes)
    angles = catersian_to_sphere(xyz)

    if args["--dump"] is not None:
        os.makedirs(args["--dump"], exist_ok=True)

    # demo: use pretrained models on NYU to predict VPs from a given image
    print('processing: ', args["<image>"])
    image_name = args["<image>"]
    image = skimage.io.imread(image_name)

    if image.shape[0:2]!=tuple([480, 640]):
        print("warning: images resized to [480, 640]!")
        image = skimage.transform.resize(image, (480,640))
        image *= 255.0
    image = np.rollaxis(image, 2).copy()
    image = torch.from_numpy(image).float().to(device)
    targets = torch.zeros(C.io.num_nodes).float().to(device)
    input_dict = {"image": image[None],  "target": targets, "eval": True}

    with torch.no_grad():
        result = model(input_dict)
    pred = result["prediction"].cpu().numpy()[0]
    
    # Option 1:
    # a. f available: first map to camera space, and then pick up the top3;
    # b. Assumption: VPs are more or less equally spread over the sphere.
    # vpts_pd, vpts_idx = topk_orthogonal_vps(pred, xyz, num_vps=3)
    
    # Option 2 - unknown f: Use clustering to detect multiple VPs
    vpts_pd, vpts_idx = vps_clustering(pred, xyz, threshold=0.5)
    angles_pd = catersian_to_sphere(vpts_pd)
    
    # You might want to resize VPs from [480, 640] to original size [img_h, img_w].
    ys, xs = to_pixel(vpts_pd, focal_length=1.0, h=480, w=640)

    ### save predictions,
    image = image.permute(1,2,0).cpu().numpy()
    if args["--dump"]:
        np.savez(
            os.path.join(args["--dump"], image_name.replace(".jpg", ".npz")),
            image = image,
            vpts_pd=vpts_pd,
            vpts_sphere=pred,
        )

    ### visualize results on the hemisphere
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(image/255.0)
    for (x, y) in zip(xs, ys):
        ax.scatter(x, y)

    ax = fig.add_subplot(122)
    ax.scatter(angles[:, 0], angles[:, 1], c=pred)
    ax.scatter(angles_pd[:, 0], angles_pd[:, 1], c='r')
    ax.set_title('Sphere')
    plt.savefig('pred.png', format='png', bbox_inches ='tight', pad_inches = 0.1, transparent=True,  dpi=600)
    plt.suptitle('VP prediction')
    plt.show()


if __name__ == "__main__":
    main()
