import os.path as osp
import torch
import numpy as np

from vpd.models.sphere.sphere_utils import gold_spiral_sampling_patch
import scipy.spatial.distance as scipy_spatial_dist

class MultiscaleRadiusGraph(object):

    def __init__(self, num_nodes, num_neighbors, cache_file=None):
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        # Cache can be used for datasets where shape is equal for every sample (like a sphere)
        self.get_cache = cache_file is not None and osp.exists(cache_file)
        self.save_cache = cache_file is not None and not self.get_cache
        self.cache_file = cache_file

    ### compute topk neighboring nodes for each node in pos, and retrun indecies [2, topkxnum_nodes]
    def compute_topk_neighbors(self, xyz):
        edge_index = np.zeros((2, 0), dtype=int)  # initialize the edge_index

        for node_idx, (node_xyz) in enumerate(xyz):
            node_xyz = node_xyz[None, :]
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
            ### compute cosine distance: same 0, orthorgonal 1, opposite 2, dist = 1-AB/(|A||B|)
            cos_dis = scipy_spatial_dist.cdist(node_xyz, xyz, 'cosine')  # 1 x num_points
            cos_dis -= 1.0
            cos_dis *= -1.0
            cos_dis = np.abs(cos_dis).flatten() # abs(AB/(|A||B|))

            ### select topk nearby nodes for each input node
            # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
            topk_min_idx = np.argpartition(cos_dis, kth=-self.num_neighbors)[-self.num_neighbors:]
            cos_dis_topk = cos_dis[topk_min_idx]

            ### to debug: minimal distance is always 0.0, because of the self_loop
            assert cos_dis_topk.max() == 1.0
 
            cur_node = np.full(shape=(self.num_neighbors), fill_value=node_idx, dtype=int)
            cur_edge_index = np.stack([cur_node, topk_min_idx], axis=0) # [2, num_neighbors]
            edge_index = np.concatenate([edge_index, cur_edge_index], axis=1)

        return edge_index


    def __call__(self):
        if self.get_cache:
            xyz, edge_index = torch.load(self.cache_file, map_location=lambda storage, loc: storage)
        else:
            xyz = gold_spiral_sampling_patch(np.array([0, 0, 1]), alpha=np.pi/2, num_pts=self.num_nodes)
            edge_index = self.compute_topk_neighbors(xyz)
            xyz = torch.from_numpy(xyz).float()
            edge_index = torch.from_numpy(edge_index).long()

            if self.save_cache:
                with open(self.cache_file, 'wb') as f:
                    torch.save((xyz, edge_index), f)
                self.save_cache = False
                self.get_cache = True
        return xyz, edge_index
