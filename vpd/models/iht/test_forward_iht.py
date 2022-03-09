import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import random
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt

from iht_utils import hough_transform
from iht_cuda import IHT_CUDA

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


IHT_cuda = IHT_CUDA(vote_mapping_dict)
IHT_cuda = IHT_cuda.to(device)

for p in range(0,100):
    print('p:', p)
    # test_in = torch.randint(low=0, high=5,size=(batch,channels, h, w)).double()
    # test_in = torch.randn(batch, channel, h, w).double()
    # test_in = F.relu(test_in)

    test_in = torch.zeros(batch,channel, h, w).double()
    pp = torch.randint(low=0, high=h*w, size=(1,))
    test_in[0,0, pp//w, pp%w] = 1.0
    test_in = test_in.to(device)
    test_in.requires_grad=True

    ##################################################################################
    iht_cuda = IHT_cuda(test_in) 
    iht_cuda = iht_cuda.cpu().detach()

    print('pp', pp)
    inds = vote_mapping[:,1]== pp.double()
    im_inds = vote_mapping[inds, 0].long()
    im_vals = vote_mapping[inds, 2]

    im_manual = torch.zeros(rows,cols).double()
    im_manual[im_inds//cols, im_inds%cols] += im_vals

    color = torch.zeros(rows, cols, 3).double()
    color[:,:,0]= iht_cuda[0,0].gt(0.0)
    color[:,:,1]= im_manual.gt(0.0)
    print('dif', (iht_cuda[0,0]-im_manual).abs().max())
    assert torch.equal(iht_cuda[0,0], im_manual)

    # fig = plt.figure()
    # ax = fig.add_subplot(131)
    # ax.imshow(ht_cuda[0,0])
    # ax = fig.add_subplot(132)
    # ax.imshow(ht_manual)
    # ax = fig.add_subplot(133)
    # ax.imshow(ht_cuda[0,0]-ht_manual)
    # # ax.imshow(color)
    # plt.show()
    print(f'forward pass correct:{p:3d}')
