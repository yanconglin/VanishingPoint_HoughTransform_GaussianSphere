import os
import json
import math
import random
import os.path as osp
import glob
import numpy as np
import torch
import skimage.io
import skimage.transform
import numpy.linalg as LA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import scipy.spatial.distance as scipy_spatial_dist
from vpd.config import C
from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch, catersian_to_sphere, sphere_to_catesian


def compute_nn_neighbors(xyz1, xyz2):
    # # # find nearest neighbors of xyz1 in xyz2
    ### compute cosine distance: same 0, orthorgonal 1, opposite 2, dist = 1-AB/(|A||B|)
    cos_dis = scipy_spatial_dist.cdist(xyz1, xyz2, 'cosine')
    # ### map to: same 1, opposite -1, orthorgonal 0, dist = AB/(|A||B|)
    cos_dis *= -1.0
    cos_dis += 1.0
    cos_dis = np.abs(cos_dis)
    cos_dis = np.clip(cos_dis, a_min=0.0, a_max=1.0)
    max_error, max_idx = cos_dis.max(axis=1), cos_dis.argmax(axis=1)
    return max_error, max_idx


def compute_topk_neighbors_self(xyz, num_neighbors):
    cos_dis = xyz @ xyz.T
    cos_dis = 1.0 - np.abs(cos_dis)
    # # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    topk_ind = np.argpartition(cos_dis, kth=num_neighbors, axis=1)[:, 0:num_neighbors]  # [N, num_neighbors]
    cur_node = np.arange(0, len(xyz)).repeat(num_neighbors)  # [N*num_neighbors]
    edges = np.stack([cur_node, topk_ind.flatten()], axis=0)
    # print('edges', edges.shape)

    return edges


def compute_scale_first(xyz_in, theta, num_nodes, xyz_ref):
    inds_scale = []
    for i, xyz_i in enumerate(xyz_in):
        print('i', i, len(xyz_in))
        xyz_j = gold_spiral_sampling_patch(xyz_i, alpha=theta * np.pi / 180., num_pts=num_nodes)
        _, inds_j = compute_nn_neighbors(xyz_j, xyz_ref)  # [num_nodes]
        inds_scale.append(inds_j)
    inds_scale = np.stack(inds_scale)
    print('compute_scale_first', inds_scale.shape)
    return inds_scale


def compute_scale_last(xyz, num_nodes):
    inds_scale = []
    for j, xyz_j in enumerate(xyz):
        print('j', j, len(xyz))
        cos_dis = xyz @ xyz_j[:, None]
        cos_dis = np.abs(cos_dis).flatten()  # [N]
        cos_dis = np.clip(cos_dis, a_min=0.0, a_max=1.0)
        cos_dis *= -1
        cos_dis += 1  
        ########################################################################################
        ### select topk nearby nodes for each input node
        # # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        inds_j = np.argpartition(cos_dis, kth=num_nodes)[0:num_nodes]
        # print('inds_j', inds_j.shape)
        inds_scale.append(inds_j)
    inds_scale = np.stack(inds_scale)
    print('compute_scale_last', inds_scale.shape)
    return inds_scale


def compute_scale_init(xyz_init, xyz):
    _, inds_init = compute_nn_neighbors(xyz_init, xyz)  # [N]
    return inds_init



class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        self.split = split
        print(self.rootdir, self.split)
        filelist = sorted(glob.glob(f"{rootdir}/*/*_0.png"))
        print("total number of samples", len(filelist))

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

        self.xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.total_nodes)
        self.angles = catersian_to_sphere(self.xyz)

        fname = f'cache/inds_{C.io.total_nodes:d}.npz'
        if os.path.isfile(fname):
            print(f'load:' + fname)
            npz = np.load(fname)
            # self.xyz = npz["xyz"]
            self.ind00 = npz["ind00"]
            self.ind0 = npz["ind0"]
            self.ind1 = npz["ind1"]
            self.ind2 = npz["ind2"]
            print('ind_scale', self.ind00.shape, self.ind0.shape, self.ind1.shape, self.ind2.shape)
        else:
            # It may take up to 2 hours to compute the index matrices for efficient sampling.
            # You can simply download the pre-calculated npz file and place it into the 'cache' folder.
            os.makedirs('cache', exist_ok=True)
            xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90.0 * np.pi / 180., num_pts=C.io.total_nodes)
            xyz_00 = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=90. * np.pi / 180., num_pts=C.io.num_nodes[0])
            ind00 = compute_scale_init(xyz_00, xyz)

            print('#### scale 0')
            assert C.io.thetas[1] > C.io.theta
            ind0 = compute_scale_first(xyz, C.io.thetas[0], C.io.num_nodes[0], xyz)  # [total, num_nodes]

            print('#### scale 1')
            assert C.io.thetas[2] > C.io.theta
            ind1 = compute_scale_first(xyz, C.io.thetas[1], C.io.num_nodes[1], xyz)  # [total, num_nodes]

            print('#### scale 2')
            assert C.io.thetas[3] < C.io.theta
            ind2 = compute_scale_last(xyz, C.io.num_nodes[2])  # [total, num_nodes]
            np.savez(f'cache/inds_{C.io.total_nodes:d}', num_nodes=C.io.num_nodes, thetas=C.io.thetas, theta=C.io.theta, xyz=xyz,
                     ind00=ind00, ind0=ind0, ind1=ind1, ind2=ind2)
            self.ind00 = ind00
            self.ind0 = ind0
            self.ind1 = ind1
            self.ind2 = ind2
            print('ind_scale', self.ind00.shape, self.ind0.shape, self.ind1.shape, self.ind2.shape)

    def __len__(self):
        return self.size

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
            if vp[2]<0.0: vp *= -1.0
            vpts.append(vp)
            # plt.scatter(
            #     vpt[0] / vpt[2] * 2.1875 * 256 + 256,
            #     -vpt[1] / vpt[2] * 2.1875 * 256 + 256
            # )
        # plt.show()
        vpts = np.array(vpts)
        input_dict = {
            "image": torch.tensor(image).float(),
            "vpts": torch.tensor(vpts).float(),
            "xyz": self.xyz,
        }

        # print('##################scale 0####################################')
        if self.split != "test":
            ### random anchor
            k = np.random.randint(0, len(self.xyz))
            ind0_graph = self.ind0[k]
        else:
            ind0_graph = self.ind00
        xyz0 = self.xyz[ind0_graph]
        angles0 = catersian_to_sphere(xyz0)
        dis_temp, ind_temp = compute_nn_neighbors(vpts, xyz0)  # [num_vpts]
        target0 = np.zeros((C.io.num_nodes[0]), dtype=np.float32)
        target0[ind_temp] = 1.0
        edge0 = compute_topk_neighbors_self(xyz0, C.io.num_neighbors[0])  # [2, num_nodes*num_neighbors]
        ind0 = ind0_graph[ind_temp]
        dis0 = dis_temp

        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # ax.scatter(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], c=target0)
        # ax.scatter(self.xyz[ind0.flatten(), 0], self.xyz[ind0.flatten(), 1], self.xyz[ind0.flatten(), 2], c='m')
        # ax.scatter(vpts[:,0], vpts[:, 1], vpts[:,2], c='r')
        # ax = fig.add_subplot(122)
        # ax.scatter(angles0[:, 0], angles0[:, 1], c='b')
        # ax.scatter(angles0[:,0], angles0[:, 1], c=target0)
        # ax.scatter(self.angles[ind0.flatten(), 0], self.angles[ind0.flatten(), 1], c='g')
        # ax.scatter(vpts_angles[:,0], vpts_angles[:, 1], c='r')
        # fig.suptitle('graphs0')
        # plt.show()
        # plt.close()

        #### scale 1
        # print('##################scale 1####################################')
        target1 = []
        ind1_graph = []
        edge1 = []
        ind1 = []
        dis1 = []
        # print('ind1', ind1.shape)
        for k, (vpt_k, dis_k, ind_k) in enumerate(zip(vpts, dis0, ind0)):  # enumerate over num_vpts
            ind1_k = self.ind1[ind_k]
            ind1_graph.append(ind1_k)
            xyz1_k = self.xyz[ind1_k]
            dis_temp, ind_temp = compute_nn_neighbors(vpt_k[None], xyz1_k)
            target1_k = np.zeros((C.io.num_nodes[1]), dtype=np.float32)
            target1_k[ind_temp] = 1.0
            target1.append(target1_k)
            edge1_k = compute_topk_neighbors_self(xyz1_k, C.io.num_neighbors[1])  # [2, num_nodes*num_neighbors]
            edge1.append(edge1_k)
            ind1.append(ind1_k[ind_temp])
            dis1.append(dis_temp)

        target1 = np.stack(target1).reshape(C.io.num_vpts, -1)  # [num_vpts, num_nodes]
        edge1 = np.stack(edge1).reshape(C.io.num_vpts, 2, -1)  # [num_vpts, 2, num_nods*num_neighbors]
        ind1_graph = np.stack(ind1_graph).reshape(C.io.num_vpts, C.io.num_nodes[1])  # [num_vpts, num_nodes]
        ind1 = np.stack(ind1).flatten()  # [num_vpts]
        dis1 = np.stack(dis1).flatten()  # [num_vpts]
        # print('target1, ind1, edge1, ind1_graph, dis1', target1.shape, ind1.shape, edge1.shape, ind1_graph.shape, dis1.shape)

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.scatter(self.angles[:, 0], self.angles[:, 1])
        # ax.scatter(self.angles[ind1_graph.flatten(), 0], self.angles[ind1_graph.flatten(), 1], c='g')
        # ax.scatter(self.angles[ind1, 0], self.angles[ind1, 1], c='m')
        # ax.scatter(vpts_angles[:,0], vpts_angles[:, 1], c='r')
        # ax = fig.add_subplot(122, projection='3d')
        # ax.scatter(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2])
        # # ax.scatter(self.xyz[ind1_graph.flatten(), 0], self.xyz[ind1_graph.flatten(), 1], self.xyz[ind1_graph.flatten(), 2], c='g',zorder=2)
        # # ax.plot3D(self.xyz[ind1_graph.flatten(), 0], self.xyz[ind1_graph.flatten(), 1], self.xyz[ind1_graph.flatten(), 2], c='g',zorder=10, lw=0, markersize=12)
        # for t in ind1.flatten():
        #     # ax.plot3D(self.xyz[t:t+1, 0], self.xyz[t:t+1, 1], self.xyz[t:t+1, 2], c='g',zorder=10)
        #     ax.scatter(self.xyz[t, 0], self.xyz[t, 1], self.xyz[t, 2], c='g', )
        # plt.suptitle('graphs1')
        # plt.show()
        # plt.close()

        #### scale 2
        target2 = []
        ind2_graph = []
        edge2 = []
        ind2 = []
        dis2 = []
        for k, (vpt_k, dis_k, ind_k) in enumerate(zip(vpts, dis1, ind1)):  # enumerate over num_vpts
            ind2_k = self.ind2[ind_k]
            ind2_graph.append(ind2_k)
            xyz2_k = self.xyz[ind2_k]
            dis_temp, ind_temp = compute_nn_neighbors(vpt_k[None], xyz2_k)
            target2_k = np.zeros((C.io.num_nodes[2]), dtype=np.float32)
            target2_k[ind_temp] = 1.0
            target2.append(target2_k)
            edge2_k = compute_topk_neighbors_self(xyz2_k, C.io.num_neighbors[2])  # [2, num_nodes*num_neighbors]
            edge2.append(edge2_k)
            ind2.append(ind2_k[ind_temp])
            dis2.append(dis_temp)

        target2 = np.stack(target2).reshape(C.io.num_vpts, -1)  # [num_vpts, num_nodes]
        edge2 = np.stack(edge2).reshape(C.io.num_vpts, 2, -1)  # [num_vpts, 2, num_nods*num_neighbors]
        ind2_graph = np.stack(ind2_graph).reshape(C.io.num_vpts, C.io.num_nodes[2])  # [num_vpts, num_nodes]
        ind2 = np.stack(ind2).flatten()  # [num_vpts]
        dis2 = np.stack(dis2).flatten()  # [num_vpts]
        # print('target2, ind2, edge2, ind2_graph, dis2', target2.shape, ind2.shape, edge2.shape, ind2_graph.shape, dis2.shape)

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.scatter(self.angles[:, 0], self.angles[:, 1])
        # ax.scatter(self.angles[ind2_graph.flatten(), 0], self.angles[ind2_graph.flatten(), 1], c='g')
        # ax.scatter(self.angles[ind2.flatten(), 0], self.angles[ind2.flatten(), 1], c='r')
        # ax = fig.add_subplot(122, projection='3d')
        # # ax.scatter(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2])
        # # ax.scatter(self.xyz[ind1.flatten(), 0], self.xyz[ind1.flatten(), 1], self.xyz[ind1.flatten(), 2], c='g',zorder=2)
        # ax.plot3D(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2])
        # ax.plot3D(self.xyz[ind2_graph.flatten(), 0], self.xyz[ind2_graph.flatten(), 1], self.xyz[ind2_graph.flatten(), 2], c='g',zorder=10)
        # fig.suptitle('graphs2')
        # plt.show()
        # plt.close()

        if self.split != "test":
            input_dict["target0"] = torch.tensor(target0).float()
            input_dict["target1"] = torch.tensor(target1).float()
            input_dict["target2"] = torch.tensor(target2).float()
            input_dict["ind0"] = torch.tensor(ind0_graph).long()
            input_dict["ind1"] = torch.tensor(ind1_graph).long()
            input_dict["ind2"] = torch.tensor(ind2_graph).long()
            input_dict["edge0"] = torch.tensor(edge0).long()
            input_dict["edge1"] = torch.tensor(edge1).long()
            input_dict["edge2"] = torch.tensor(edge2).long()
            # for k, v in input_dict.items():
            #     print(k, v.shape)
            return input_dict
        else:
            input_dict["xyz"] = torch.tensor(self.xyz).float()
            input_dict["edge0"] = torch.tensor(edge0).long()

            input_dict["ind0"] = torch.tensor(self.ind00).long()
            input_dict["ind1"] = torch.tensor(self.ind1).long()
            input_dict["ind2"] = torch.tensor(self.ind2).long()

            input_dict["target0"] = torch.tensor(target0).float()
            input_dict["target1"] = torch.tensor(target1).float()
            input_dict["target2"] = torch.tensor(target2).float()
            return input_dict


if __name__ == "__main__":
    C.update(C.from_yaml(filename="config/su3.yaml"))
    dataset = WireframeDataset(C.io.datadir, split="test")
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=16)

    for k, input_dict in enumerate(data_loader):
        for i, v in input_dict.items():
            print(i, v.shape)
        print("total", k)
