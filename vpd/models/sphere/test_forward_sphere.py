import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import random
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from sphere_cuda import SPHERE_CUDA

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


if torch.cuda.is_available():
    # device_name = "cuda"
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed(0)

    device_name = "cuda"
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.enabled = False  #greatly slow down the speed
    torch.backends.cudnn.deterministic = True
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
else:
    device_name = "cpu"
    print("CUDA is not available")
device = torch.device(device_name)


batch = 2
channel = 4
rows = 64
cols = 64
h = 93
w = 180


npz_name = f"vpd/vote_mapping/grad_test/sphere_neighbors_weight_93_180_4096_nn_rearrange.npz"

npzfile = np.load(npz_name, allow_pickle=True)
print('npzfile', npzfile.files)
vote_mapping_sphere = npzfile['sphere_neighbors_weight']
sphere_size= npzfile['num_points']
print('vote_mapping_sphere',vote_mapping_sphere.shape)
num_pts, num_votes, _ = vote_mapping_sphere.shape

# for grad test, use double instead of float
vote_mapping_sphere = torch.from_numpy(vote_mapping_sphere).double().contiguous()
print('vote_mapping_sphere  memory MB', vote_mapping_sphere.size(), vote_mapping_sphere.element_size() * vote_mapping_sphere.nelement() / (1024 * 1024))

vote_mapping_dict={}
vote_mapping_sphere = vote_mapping_sphere.to(device)
vote_mapping_dict["vote_mapping"] = vote_mapping_sphere
vote_mapping_dict["sphere_size"] = sphere_size
vote_mapping_dict["ht_size"] = (npzfile['h'], npzfile['w'])
Sphere_cuda = SPHERE_CUDA(vote_mapping_dict)

Sphere_cuda = Sphere_cuda.to(device)

for p in range(0,10):
    print('p:', p)
    test_in = torch.randint(low=0, high=5,size=(batch, channel, rows, cols)).double()
    test_in = torch.randn(batch, channel, h, w).double()
    test_in = F.relu(test_in)
    # test_in = torch.zeros(batch,channel, h, w).double()
    # pp = torch.randint(low=0, high=h*w, size=(1,))
    # test_in[0,0,pp//w,pp%w] = 1.0
    test_in = test_in.to(device)
    test_in.requires_grad=True

    inds = torch.randint(0, 4096, size=(batch, 128)).to(device)
    ##################################################################################
    sphere_cuda = Sphere_cuda(test_in, inds) 
    sphere_cuda = sphere_cuda.cpu().detach()
    print('sphere_cuda:', test_in.shape, sphere_cuda.shape)

    ##################################################################################
    print(f'forward pass correct:{p:3d}')
