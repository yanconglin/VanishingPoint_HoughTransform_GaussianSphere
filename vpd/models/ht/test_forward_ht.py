import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import random
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from ht_cuda import HT_CUDA


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

HT_cuda = HT_CUDA(vote_mapping_dict)
HT_cuda = HT_cuda.to(device)


for p in range(0,100):
    print('p:', p)
    # test_in = torch.randint(low=0, high=5,size=(batch,channels, rows, cols)).double()
    # test_in = torch.randn(batch, channel, h, w).double()
    # test_in = F.relu(test_in)
    test_in = torch.zeros(batch,channel, rows, cols).double()
    pp = torch.randint(low=0, high=rows*cols, size=(1,))
    test_in[0,0, pp//cols, pp%cols] = 1.0
    test_in = test_in.to(device)
    test_in.requires_grad=True

    ##################################################################################
    ht_cuda = HT_cuda(test_in) 
    ht_cuda = ht_cuda.cpu().detach()
    print('ht_cuda:', test_in.shape, ht_cuda.shape)

    ##################################################################################
    print('pp', pp)
    inds = vote_mapping[:, 0]== pp.double()
    ht_inds = vote_mapping[inds, 1].long()
    ht_vals = vote_mapping[inds, 2]
    print('ht_vals', ht_inds.sum(), ht_vals.sum())

    ht_mannual = torch.zeros(h,w).double()
    ht_mannual[ht_inds//w, ht_inds%w] += ht_vals
    # ht_mannual /= np.float64(cols)
    # for ht_ind, ht_val in zip(ht_inds,ht_vals):
    #     ht_mannual[ht_ind//w, ht_ind%w] += ht_val
    #     print(f'ht_inds:{ht_ind:04d}, ht_h:{ht_ind//w:03d}, ht_w:{ht_ind%w:03d}, ht_val:{ht_val:.8f}')

    color = torch.zeros(h,w, 3).double()
    color[:,:,0]= ht_cuda[0,0].gt(0.0)
    color[:,:,1]= ht_mannual.gt(0.0)
    print('dif', (ht_cuda[0,0]-ht_mannual).abs().max())
    assert torch.equal(ht_cuda[0,0], ht_mannual)

    # fig = plt.figure()
    # ax = fig.add_subplot(131)
    # ax.imshow(ht_cuda[0,0])
    # ax = fig.add_subplot(132)
    # ax.imshow(ht_mannual)
    # ax = fig.add_subplot(133)
    # ax.imshow(ht_cuda[0,0]-ht_mannual)
    # # ax.imshow(color)
    # plt.show()
    print(f'forward pass correct:{p:3d}')
