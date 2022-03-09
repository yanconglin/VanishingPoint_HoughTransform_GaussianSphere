#!/usr/bin/env python3
"""Detect vanishing points with the Manhattan assumption.
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

import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import numpy.linalg as LA
import scipy.spatial.distance as scipy_spatial_dist
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from tqdm import tqdm
from docopt import docopt
import scipy.io as sio
import vpd
from vpd.config import C, M
from vpd.datasets import WireframeDataset

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
            # print('detected vp point (cos, cos_arc, cos_angle)', dist_cos, dist_cos_arc, dist_cos_angle)
            vps_idx.append(i)
        else:
            continue

    vps_pd = xyz[vps_idx]
    return vps_pd, vps_idx

def compute_error(vps_pd, vps_gaussian):
    # print('error', vps_pd.shape, vps_gaussian.shape)
    error = np.arccos(np.abs(vps_gaussian @ vps_pd.transpose()).clip(max=1))
    error = error.min(axis=1) / np.pi * 180.0 # num_pd x num_gt, axis=1
    
    # error = np.arccos(np.abs(vps_pd @ vps_gaussian.transpose()).clip(max=1))
    # error = error.min(axis=1) / np.pi * 180.0 # num_pd x num_gt, axis=1
    # error = scipy_spatial_dist.cdist(vps_pd, vps_gaussian, 'cosine')
    # error *= -1
    # error += 1
    # error = np.arccos(np.abs(error))
    #
    # error = error.min(axis=1) / np.pi * 180.0 # num_pd x num_gt, axis=1
    # # print('error', error.shape, error)
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
        use_gpu=True
        device_name = "cuda"
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        for k in range(0, torch.cuda.device_count()):
            print('kth, device name', k, torch.cuda.get_device_name(k))
    else:
        use_gpu=False
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
    # check_dir('cache')
    if M.backbone == "stacked_hourglass":
        backbone = vpd.models.hg(
            planes=128, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
        )
    else:
        raise NotImplementedError

    # model = vpd.models.VanishingNet(backbone, vote_ht_dict, vote_sphere_dict)
    model = vpd.models.VanishingNet_test(backbone, vote_ht_dict, vote_sphere_dict)   
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
    # print('model', model)
    ##### number of parameters in a model
    total_params = sum(p.numel() for p in model.parameters())
    ##### number of trainable parameters in a model
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of total parameters', total_params)
    print('num of trainable parameters', train_params)


    loader = torch.utils.data.DataLoader(
        WireframeDataset(C.io.datadir, split="test"),
        batch_size=1,
        shuffle=False,
        # num_workers=C.io.num_workers if os.name != "nt" else 0,
        num_workers=2,
        pin_memory=True,
    )
    print('loader size',  len(loader))
    if args["--dump"] is not None:
        os.makedirs(args["--dump"], exist_ok=True)

    errors = []
   

    for batch_idx, input_dict in enumerate(tqdm(loader)):
        input_dict["eval"]=True

        with torch.no_grad():
            result = model(input_dict)
            
        vpts1 = result["vpts1"].cpu().numpy()
        vpts2 = result["vpts2"].cpu().numpy()  
        vpts3 = result["vpts3"].cpu().numpy()  
               
        vpts_gt = input_dict["vpts"].cpu().numpy()[0]
        ind0 = input_dict["ind0"].cpu().numpy()[0]
                
        target0 = input_dict["target0"].cpu().numpy()[0]
        target1 = input_dict["target1"].cpu().numpy()[0]
        target2 = input_dict["target2"].cpu().numpy()[0]
            
        error = compute_error(vpts3, vpts_gt)
        errors.append(error)
        # print('error', error.shape, error, vps_pd.shape, gt_vpt.shape)
        
        pred0 = result["pred0"].cpu().numpy()
        pred1 = result["pred1"].cpu().numpy()
        pred2 = result["pred2"].cpu().numpy()
        
        idx1 = result["idx1"].cpu().numpy()
        anchor1 = result["anchor1"].cpu().numpy()
        ind1 = result["ind1"].cpu().numpy()
        
        idx2 = result["idx2"].cpu().numpy()
        anchor2 = result["anchor2"].cpu().numpy()
        ind2 = result["ind2"].cpu().numpy()
        
        idx3 = result["idx3"].cpu().numpy()
        anchor3 = result["anchor3"].cpu().numpy()

        # # ### visualize in angles
        # # # ### visualize in angles
        # fig = plt.figure()
        # ax = fig.add_subplot(131)
        # ax.scatter(angles[:, 0], angles[:, 1], c=target.flatten())
        # ax.set_title('gt')
        #
        # ax = fig.add_subplot(132)
        # ax.scatter(angles[:, 0], angles[:, 1], c=pred.flatten())
        # ax.set_title('pred '+ str(pred.max().item()))
        #
        # ax = fig.add_subplot(133)
        # # ax.scatter(angles[:,0], angles[:,1], c=data.x.squeeze(1))
        # color = np.zeros((num_nodes, 3))
        # color[:, 0] = target.flatten()
        # # color[:, 2] = pred.flatten()
        # color[:,2] = pred_topk.flatten()
        # ax.scatter(angles[:, 0], angles[:, 1], c=color)
        # for (vp_idx, err) in zip(vps_idx, error):
        #         ax.annotate(f"{err.item():.02f}", xy=(angles[vp_idx, 0], angles[vp_idx, 1]), xytext=(angles[vp_idx, 0]+0.1, angles[vp_idx, 1]+0.1), c='w')
        # ax.set_title('r-gt, b-pred')
        # plt.suptitle('Gaussian sphere-fibonacci ' + str(batch_idx))
        # plt.show()
        
        ### save predictions, gt and errors
        if args["--dump"]:
            index = batch_idx
            np.savez(
                os.path.join(args["--dump"], f"{index:06d}.npz"),
                vpts1=vpts1,
                vpts2=vpts2,
                vpts3=vpts3,
                
                ind0 = ind0,
                vpts_gt=vpts_gt,
                err=error,
                
                pred0=pred0,
                pred1=pred1,
                pred2=pred2,

                target0=target0,
                target1=target1,
                target2=target2,
                
                idx1=idx1,
                anchor1 = anchor1, 
                ind1 = ind1,
                
                idx2=idx2,
                anchor2 = anchor2, 
                ind2 = ind2,
                
                idx3=idx3,
                anchor3 = anchor3, 
            )



    if args["--output"]:
    	np.savez(os.path.join(args["--output"]), err=np.vstack(errors))
    errors = np.sort(np.hstack(errors))
    y = (1 + np.arange(len(errors))) / len(errors)


    if not args["--noimshow"]:
        plt.plot(errors, y, label="Conic")
        print(" | ".join([f"{AA(errors, y, th):.3f}" for th in [1, 3, 5, 10]]))
        plt.xlim(0.0, 90.0, 1.0)
        plt.ylim(0.0, 1.0, 0.1)
        plt.legend()
        # plt.savefig('errors.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0.1, transparent=True,  dpi=600)
        plt.show()


if __name__ == "__main__":
    main()
