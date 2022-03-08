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


batch = 2
channel = 4
rows = 48
cols = 48
h = 69
w = 60
theta_res = 3
rho_res = 1
num_points=128
sphere_size=512


npz_name = f"vote_{rows:d}_{cols:d}_{h:d}_{w:d}_{num_points:d}_{sphere_size:d}.npz"

npzfile = np.load(npz_name, allow_pickle=True)
print('npzfile', npzfile.files)
vote_mapping_sphere = npzfile['vote_mapping_sphere']
sphere_size= npzfile['sphere_size']
angles= npzfile['angles']
xyz= npzfile['xyz']
print('vote_mapping_sphere',vote_mapping_sphere.shape)

# for grad test, use double instead of float
vote_mapping_sphere = torch.from_numpy(vote_mapping_sphere).double().contiguous()
print('vote_mapping_sphere  memory MB', vote_mapping_sphere.size(), vote_mapping_sphere.element_size() * vote_mapping_sphere.nelement() / (1024 * 1024))

vote_mapping_dict={}
vote_mapping_dict["vote_mapping"] = vote_mapping_sphere.to(device)
vote_mapping_dict["sphere_size"] = sphere_size
vote_mapping_dict["ht_size"] = [h, w]
Sphere_cuda = SPHERE_CUDA(vote_mapping_dict)

Sphere_cuda = Sphere_cuda.to(device)

print('grad check***********')

input = torch.randn(batch, channel, h, w, requires_grad=True).double().to(device)
res = gradcheck(Sphere_cuda, input, raise_exception=True)
# res=gradcheck(myconv, input, eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True)
print('grad check', res)
