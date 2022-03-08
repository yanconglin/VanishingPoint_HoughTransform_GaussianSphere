import os
import json
import math
import random
import glob
import numpy as np
import torch
from scipy import io
import skimage.io
import numpy.linalg as LA
import matplotlib.pyplot as plt
import skimage.transform
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import scipy.spatial.distance as scipy_spatial_dist
from vpd.config import C
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = sorted(glob.glob(f"{rootdir}/*/*_0.png"))
        print("total number of samples", len(filelist))

        self.split = split
        division = int(len(filelist) * 0.1)
        print("num of valid/test", division)
        if split == "train":
            num_train = int(len(filelist) * 0.8 * C.io.percentage)
            self.filelist = filelist[2 * division: 2 * division + num_train]
            self.size = len(self.filelist)
            print("subset for training: percentage ", C.io.percentage, num_train)
        if split == "valid":
            self.filelist = [f for f in filelist[division:division*2] if "a1" not in f]
            self.size = len(self.filelist)
        if split == "test":
            self.filelist = [f for f in filelist[:division] if "a1" not in f]
            self.size = len(self.filelist)
        print(f"n{split}:", len(self.filelist))

        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.num_nodes)

    def __len__(self):
        return self.size

    def cos_cdis(self, x, y, semi_sphere=False):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        ### compute cosine distance
        ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
        dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  
        dist_cos *= -1.0
        dist_cos += 1.0

        if semi_sphere is True: dist_cos = np.abs(dist_cos)  # dist = abs|AB/(|A||B|)|
        dist_cos_arc = np.arccos(dist_cos)
        return dist_cos_arc

    def __getitem__(self, idx):

        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname).astype(float)[:, :, 0:3]
        image = np.rollaxis(image, 2).copy()
        prefix = iname.replace(".png", "")
        with open(f"{prefix}_camera.json") as f:
            js = json.load(f)
            RT = np.array(js["modelview_matrix"])

        vpts = []
        # plt.imshow(io.imread(iname))
        for axis in [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]:
            vp = RT @ axis
            vp = np.array([vp[0], vp[1], -vp[2]])
            vp /= LA.norm(vp)
            if vp[2] < 0.0: vp *= -1.0
            vpts.append(vp)
            # plt.scatter(
            #     vpt[0] / vpt[2] * 2.1875 * 256 + 256,
            #     -vpt[1] / vpt[2] * 2.1875 * 256 + 256
            # )
        # plt.show()
        vpts = np.array(vpts)
        dis = self.cos_cdis(vpts, self.xyz)
        vpts_idx = dis.argmin(axis=1)
        label = np.zeros((C.io.num_nodes), dtype=np.float32)
        label[vpts_idx] = 1.0

        return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.tensor(vpts).float()


class ScanNetDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split
        print(self.rootdir, self.split)
        dirs = np.genfromtxt(f"{rootdir}/scannetv2_{split}.txt", dtype=str)
        filelist = sum([sorted(glob.glob(f"{rootdir}/{d}/*.png")) for d in dirs], [])
        print("total number of samples", len(filelist))

        if split == "train":
            num_train = int(len(filelist) * C.io.percentage)
            self.filelist = filelist[0 : num_train]
            self.size = len(self.filelist)
        if split == "valid":
            random.seed(0)
            random.shuffle(filelist)
            self.filelist = filelist[:500]
            self.size = len(self.filelist)
        if split == "test":
            random.seed(0)
            random.shuffle(filelist)
            self.filelist = filelist[:2000] # randomly sample 2k images for a quick test.
            # self.filelist = filelist # all test images (~20k).
            self.size = len(self.filelist)
        print(f"n{split}:", self.size)

        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.num_nodes)

    def __len__(self):
        return self.size

    def cos_cdis(self, x, y, semi_sphere=False):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        ### compute cosine distance
        ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
        dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  
        dist_cos *= -1.0
        dist_cos += 1.0

        if semi_sphere is True: dist_cos = np.abs(dist_cos)  # dist = abs|AB/(|A||B|)|
        dist_cos_arc = np.arccos(dist_cos)
        return dist_cos_arc

    def __getitem__(self, idx):

        iname = self.filelist[idx % len(self.filelist)]
        image = skimage.io.imread(iname)[:, :, 0:3]
        image = np.rollaxis(image, 2).copy().astype(float)

        with np.load(iname.replace("color.png", "vanish.npz")) as npz:
            vpts = np.array([npz[d] for d in ["x", "y", "z"]])
        vpts[:, 1] *= -1
        vpts /= LA.norm(vpts, axis=1, keepdims=True)

        dist_cos_arc = self.cos_cdis(vpts, self.xyz, semi_sphere=True)  # num_points_ x num_nodes
        vpts_idx = dist_cos_arc.argmin(axis=1)
        label = np.zeros((C.io.num_nodes), dtype=np.float32)
        label[vpts_idx] = 1.0
        return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.tensor(vpts).float()


class NYUDataset(Dataset):
    def __init__(self, rootdir="dataset/nyu_vp/processed_data", split='all'):
        filelist = glob.glob(f"{rootdir}/*_0.png")
        filelist.sort()
        self.rootdir = rootdir
        self.split = split
        if split == "train":
            self.filelist = filelist[0:1000]
            # self.filelist = filelist[0:1224]
            self.size = len(self.filelist) * 4
            print("subset for training: ", self.size)

        if split == "valid":
            self.filelist = filelist[1000:1224]
            self.size = len(self.filelist)
            print("subset for valid: ", self.size)

        if split == "test":
            self.filelist = filelist[1224:1449]
            self.size = len(self.filelist)
            print("subset for test: ", self.size)

        if split == "all":
            self.filelist = filelist
            self.size = len(self.filelist)
            print("all: ", len(self.filelist))

        self.num_nodes = C.io.num_nodes
        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi/2, num_pts=C.io.num_nodes)
        
    def __len__(self):
        return self.size


    def cos_cdis(self, x, y, semi_sphere=False):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        ### compute cosine distance
        ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
        dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  
        dist_cos *= -1.0
        dist_cos += 1.0

        if semi_sphere is True: dist_cos = np.abs(dist_cos)  # dist = abs|AB/(|A||B|)|
        dist_cos_arc = np.arccos(dist_cos)
        return dist_cos_arc

    def __getitem__(self, idx):
        
        if self.split == "train":
            iname = self.filelist[idx//4]
            if idx%4==0: iname=iname
            if idx%4==1: iname=iname.replace("_0", "_1")
            if idx%4==2: iname=iname.replace("_0", "_2")
            if idx%4==3: iname=iname.replace("_0", "_3")
        else:
            iname = self.filelist[idx]

        image = skimage.io.imread(iname)[:, :, 0:3]
        image = np.rollaxis(image, 2).copy().astype(float)

        with np.load(iname.replace(".png", ".npz"), allow_pickle=True) as npz:
            vpts = npz["vpts"]
            
        dist_cos = self.cos_cdis(vpts, self.xyz, semi_sphere=True)  # num_points_ x num_nodes
        vpts_idx = dist_cos.argmin(axis=1)
        label = np.zeros((C.io.num_nodes), dtype=np.float32)
        label[vpts_idx] = 1.0

        # return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.tensor(vpts).float()
        return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.zeros(3,3).float()

### YUD is only used for cross-dataset test. Need to resize the image and recalculate vps once focal length differs.
class YUDDataset(Dataset):
    def __init__(self, rootdir="dataset/YorkUrbanDB/yud_plus/data/processed_data_aug", split='all', yudplus=False):
        filelist = glob.glob(f"{rootdir}/*_0.png")
        filelist.sort()
        self.rootdir = rootdir
        self.split = split
        self.yudplus = yudplus
        if split == "train":
            self.filelist = filelist[0:25]
            self.size = len(self.filelist) * 4
            print("subset for training: ", self.size)

        if split == "valid":
            self.filelist = filelist[25:102]
            self.size = len(self.filelist)
            print("subset for valid: ", self.size)

        if split == "test":
            self.filelist = filelist[25:102]
            self.size = len(self.filelist)
            print("subset for test: ", self.size)

        if split == "all":
            self.filelist = filelist
            self.size = len(self.filelist)
            print("all: ", len(self.filelist))

        camera_params = io.loadmat(os.path.join("dataset/YorkUrbanDB/yud_plus/data/YorkUrbanDB", "cameraParameters.mat"))
        f = camera_params['focal'][0, 0]
        ps = camera_params['pixelSize'][0, 0]
        pp = camera_params['pp'][0, :]
        K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        S = np.matrix([[2.0 / 640, 0, -1], [0, 2.0 / 640, -0.75], [0, 0, 1]])
        invmat = np.linalg.inv(S * K)
        self.invmat = np.array(invmat)

        self.num_nodes = C.io.num_nodes
        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi/2, num_pts=C.io.num_nodes)
        
    def cos_cdis(self, x, y, semi_sphere=False):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        ### compute cosine distance
        ### scipy: same 0, opposite 2, orthorgonal 1, dist = 1-AB/(|A||B|)
        dist_cos = scipy_spatial_dist.cdist(x, y, 'cosine')  
        dist_cos *= -1.0
        dist_cos += 1.0

        if semi_sphere is True: dist_cos = np.abs(dist_cos)  # dist = abs|AB/(|A||B|)|
        dist_cos_arc = np.arccos(dist_cos)
        return dist_cos_arc


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.split == "train":
            iname = self.filelist[idx // 4]
            if idx % 4 == 0: iname = iname
            if idx % 4 == 1: iname = iname.replace("_0", "_1")
            if idx % 4 == 2: iname = iname.replace("_0", "_2")
            if idx % 4 == 3: iname = iname.replace("_0", "_3")
        else:
            iname = self.filelist[idx]

        image = skimage.io.imread(iname)[:, :, 0:3]
        image = np.rollaxis(image, 2).copy().astype(float)

        with np.load(iname.replace(".png", "_label_" + str(C.io.num_nodes) + ".npz"), allow_pickle=True) as npz:
            vpts = npz["vpts"]

        if not self.yudplus:
            vpts = vpts[0:3]

        dist_cos_arc = self.cos_cdis(vpts, self.xyz, semi_sphere=True)
        vpts_idx = dist_cos_arc.argmin(axis=1)
        
        label = np.zeros((C.io.num_nodes), dtype=np.float32)
        label[vpts_idx] = 1.0
        
        return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.tensor(vpts).float()
        # return torch.tensor(image).float(), torch.from_numpy(label).float(), torch.zeros((3, 3)).float()
