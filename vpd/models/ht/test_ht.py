import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
import random
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from neurvps.models.ht.ht_torch import HT_TORCH
from neurvps.models.ht.ht_cuda import HT_CUDA


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


if torch.cuda.is_available():

    device_name = "cuda"
    torch.cuda.manual_seed(0)
    # torch.backends.cudnn.enabled = False  #greatly slow down the speed
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
else:
    device_name = "cpu"
    print("CUDA is not available")
device = torch.device(device_name)


batch =16
channel=256
rows = 128
cols = 128
theta_res = 3
rho_res = 1
h=182
w=60

# save_v_mat = f"vpd/vote_matrix_{rows:d}_{cols:d}_{theta_res:d}_{rho_res:d}.mat"
# # save_v_mat = f"../vote_matrix_{rows:d}_{cols:d}_{theta_res:d}_{rho_res:d}.mat"
#
# # vote_matrix = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
# # h, w = vote_matrix.shape[2:4]
# # sio.savemat(save_v_mat, {'vote_matrix': vote_index, "rows":rows, "cols":cols, "h":h, "w":w, "theta_res":theta_res, "rho_res":rho_res,})
# vote_matrix = sio.loadmat(save_v_mat)['vote_matrix']
# vote_matrix = vote_matrix.astype(np.float32)
# h, w = vote_matrix.shape[2:4]
# print('vote_matrix', vote_matrix.shape)

save_v_mat = f"vpd/vote_index_{rows:d}_{cols:d}_{theta_res:d}_{rho_res:d}.mat"
# save_v_mat = f"../vote_index_{rows:d}_{cols:d}_{theta_res:d}_{rho_res:d}.mat"
# vote_index = matrix2index(vote_index)
# sio.savemat(save_v_mat, {'vote_index': vote_index, "rows":rows, "cols":cols, "h":h, "w":w, "theta_res":theta_res, "rho_res":rho_res,})
vote_index = sio.loadmat(save_v_mat)['vote_index']
vote_index = vote_index.astype(np.float32)
print('vote_index', vote_index.shape)

vote_index = torch.from_numpy(vote_index).float().contiguous()
print('vote_index  memory MB', vote_index.size(), vote_index.element_size() * vote_index.nelement() / (1024 * 1024))
print('vote_index', vote_index.shape, vote_index.max(), vote_index.min())
vote_index_dict={}
vote_index_dict["vote_index"] = vote_index.to(device)
vote_index_dict["im_size"] = [rows, cols]
vote_index_dict["ht_size"] = [h, w]



model =[]
model.append(HT_CUDA(vote_index_dict))
# model.append(nn.Conv2d(channel, 1, kernel_size=1))
model = nn.Sequential(*model)
model = model.to(device)
print('model', model)

for p in range(0,100):
    print('p:', p, batch, channel, rows, cols)
    test_in = torch.randn(batch, channel, rows, cols).float()
    test_in = test_in.to(device)
    test_in = F.relu(test_in)
    test_in.requires_grad=True

    test_out = model(test_in)

    test_out.mean().backward()

