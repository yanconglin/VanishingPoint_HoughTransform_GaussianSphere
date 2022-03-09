#!/usr/bin/env python3
"""Detect vanishing points in non-Manhattan world.
Usage:
    eval.py [options] <yaml-config> <checkpoint>
    eval.py ( -h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint

Options:
   -h --help                     Show this screen
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -o --output <output>          Path to the output AA curve [default: error.npz]
   --dump <output-dir>           Optionally, save the vanishing points to npz format.
                                 The coordinate of VPs is in the camera space, see
                                 `to_label` and `to_pixel` in vpd/models/vanishing_net.py
                                 for more details.
   --noimshow                    Do not show result
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
import vpd
import vpd.models.vanishing_net as vn
from vpd.config import C, M
from vpd.datasets import ScanNetDataset, WireframeDataset, YUDDataset, NYUDataset
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch

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

def compute_error(vps_pd, vps_gt):
    error = np.arccos(np.abs(vps_gt @ vps_pd.transpose()).clip(max=1))
    error = error.min(axis=1) / np.pi * 180.0 # num_pd x num_gt, axis=1
    return error.flatten()


def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


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

    if args["<checkpoint>"] =="None":
        checkpoint = None
    else:
        print('args["<checkpoint>"]', args["<checkpoint>"])
        checkpoint = torch.load(args["<checkpoint>"], map_location=lambda storage, loc: storage)
        print('checkpoint', checkpoint["iteration"], checkpoint["epoch"])
        # print('checkpoint', checkpoint["iteration"])
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print('model', model)

    ##### number of parameters in a model
    total_params = sum(p.numel() for p in model.parameters())
    ##### number of trainable parameters in a model
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', total_params)
    print('num of trainable parameters', train_params)

    if C.io.dataset.upper() == "WIREFRAME":
        Dataset = WireframeDataset
    elif C.io.dataset.upper() == "SCANNET":
        Dataset = ScanNetDataset
    elif C.io.dataset.upper() == "NYU":
        Dataset = NYUDataset
    elif C.io.dataset.upper() == "YUD":
        Dataset = YUDDataset
    else:
        raise NotImplementedError

    # assert C.io.dataset.upper() in ["NYU", "YUD"]
    assert C.io.dataset.upper() in ["NYU"]

    loader = torch.utils.data.DataLoader(
        Dataset(C.io.datadir, split="test"),
        batch_size=M.batch_size * num_gpus,
        shuffle=False,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )
    print('loader size',  len(loader))
    if args["--dump"] is not None:
        os.makedirs(args["--dump"], exist_ok=True)

    xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.num_nodes)

    for batch_idx, (images, targets, vpts_gt) in enumerate(tqdm(loader)):
        images = images.to(device)
        targets = targets.to(device)
        input_dict = {"image": images, "target": targets, "eval": True}

        with torch.no_grad():
            result = model(input_dict)
        preds = result["prediction"].cpu().numpy()
        targets = targets.cpu().numpy()
        vpts_gt = vpts_gt.cpu().numpy()

        for idx, (pred, target, vpt_gt) in enumerate(zip(preds, targets, vpts_gt)):
            ### save predictions at first and then cluster VPs
            if args["--dump"]:
                index = batch_idx * M.batch_size + idx
                np.savez(
                    os.path.join(args["--dump"], f"{index:06d}.npz"),
                    vpts_sphere=pred,
                )

if __name__ == "__main__":
    main()
