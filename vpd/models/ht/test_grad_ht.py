import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import random
import numpy as np
import math
import scipy.io as sio

from ht_cuda import HT_CUDA
from ht_utils import hough_transform

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



batch = 1
channel = 1
rows = 48
cols = 48
theta_res = 3
rho_res = 1

# vote_mapping, rho, theta = hough_transform(rows, cols, theta_res, rho_res)
# h = len(rho)
# w = len(theta)

# ht_npz_name= f"vote_index_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
# np.savez(ht_npz_name,
#             vote_mapping=vote_mapping,
#             rho=rho,
#             theta=theta,
#             rows=rows,
#             cols=cols,
#             h=h,
#             w=w,
#             theta_res=theta_res,
#             rho_res=rho_res)

h = 69
w = 60
ht_npz_name= f"vote_index_{rows:d}_{cols:d}_{h:d}_{w:d}.npz"
ht_npzfile = np.load(ht_npz_name, allow_pickle=True)
vote_mapping = ht_npzfile['vote_mapping']
vote_mapping = vote_mapping.astype(np.float32)
rho, theta = ht_npzfile["rho"],  ht_npzfile["theta"]
print('vote_mapping', vote_mapping.shape)


# for grad test, use double instead of float
vote_mapping = torch.from_numpy(vote_mapping).double().contiguous()
print('vote_mapping  memory MB', vote_mapping.size(), vote_mapping.element_size() * vote_mapping.nelement() / (1024 * 1024))

vote_mapping_dict={}
vote_mapping_dict["vote_mapping"] = vote_mapping.to(device)
vote_mapping_dict["im_size"] = [rows, cols]
vote_mapping_dict["ht_size"] = [h, w]

HT_cuda = HT_CUDA(vote_mapping_dict)
HT_cuda = HT_cuda.to(device)


print('grad check***********')

input = torch.randn(batch, channel, rows, cols, requires_grad=True).double().to(device)
res = gradcheck(HT_cuda, input, raise_exception=True)
# res=gradcheck(myconv, input, eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True)
print('grad check', res)
