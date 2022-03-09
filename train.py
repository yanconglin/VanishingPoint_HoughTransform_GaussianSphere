#!/usr/bin/env python3
"""Training and Evaluate a Neural Network
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
    yaml-config                      Path to the yaml hyper-parameter file

Options:
   -h --help                         Show this screen.
   -d --devices <devices>            Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>      Folder name [default: default-identifier]
"""

import os
import sys
import glob
import shlex
import pprint
import random
import shutil
import signal
import os.path as osp
import datetime
import platform
import threading
import subprocess

import yaml
import numpy as np
import torch
import scipy.io as sio
from docopt import docopt
import vpd
from vpd.config import C, M
from vpd.datasets import NYUDataset, WireframeDataset, ScanNetDataset, YUDDataset

def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)

# def git_hash():
#     cmd = 'git log -n 1 --pretty="%h"'
#     ret = subprocess.check_output(shlex.split(cmd)).strip()
#     if isinstance(ret, bytes):
#         ret = ret.decode()
#     return ret


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    # os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    print('torch version', torch.__version__)
    device_name = "cpu"
    num_gpus = args["--devices"].count(",") + 1
    print('num_gpus', num_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.cuda.empty_cache()
        # https://github.com/NVIDIA/pix2pixHD/issues/176
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print('cudnn', torch.backends.cudnn.version())
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        for k in range(0, torch.cuda.device_count()):
            print('kth, device name', k, torch.cuda.get_device_name(k))
    else:
        use_gpu=False
        assert use_gpu is False, "CUDA is not available"
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    batch_size = M.batch_size * num_gpus
    datadir = C.io.datadir
    num_workers = C.io.num_workers * num_gpus
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    
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
    train_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="train"), shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(datadir, split="valid"), shuffle=False, **kwargs
    )
    epoch_size = len(train_loader)
    print('epoch_size: train/valid',  len(train_loader), len(val_loader))

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
    # print('model', model)
    ##### number of parameters in a model
    total_params = sum(p.numel() for p in model.parameters())
    print('num of total parameters', total_params)
    ##### number of trainable parameters in a model
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num of trainable parameters', train_params)
    
    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr * num_gpus,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=C.optim.lr * num_gpus,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        print('resume_from', resume_from)
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])
    outdir = resume_from or get_outdir(args["--identifier"])
    print("outdir:", outdir)

    try:
        trainer = vpd.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=batch_size,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.epoch = checkpoint["epoch"]
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            print('trainer.epoch, trainer.iteration, trainer.best_mean_loss ', trainer.epoch, trainer.iteration, trainer.best_mean_loss )
            del checkpoint
        trainer.train()
        print('finish trainig at: ', str(datetime.datetime.now()))
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise


if __name__ == "__main__":
    main()
