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

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


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



batch = 1
channel = 2
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
vote_mapping_sphere.requires_grad=False
print('vote_mapping_sphere  memory MB', vote_mapping_sphere.size(), vote_mapping_sphere.element_size() * vote_mapping_sphere.nelement() / (1024 * 1024))

vote_mapping_dict={}
vote_mapping_sphere = vote_mapping_sphere.to(device)
vote_mapping_dict["vote_mapping"] = vote_mapping_sphere
vote_mapping_dict["sphere_size"] = sphere_size
vote_mapping_dict["ht_size"] = (npzfile['h'], npzfile['w'])
Sphere_cuda = SPHERE_CUDA(vote_mapping_dict).double()
Sphere_cuda = Sphere_cuda.to(device)

print('grad check***********')

x = torch.randn(batch, channel, h, w, requires_grad=True).double().to(device)
inds = torch.randint(0, 4096, size=(batch, 96), requires_grad=False).to(device)
# res = gradcheck(Sphere_cuda, (x, inds), raise_exception=True)
res =gradcheck(Sphere_cuda, (x, inds), eps=1e-4, raise_exception=True)
# res =gradcheck(Sphere_cuda, (x, inds), eps=1e-3, atol=1e-3, raise_exception=True)
print('grad check', res)
