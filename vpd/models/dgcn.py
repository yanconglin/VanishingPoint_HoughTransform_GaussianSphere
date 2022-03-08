import os
import torch
import torch.nn as nn
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_max
import torch_geometric.transforms as T
from torch_geometric.data import Data
from vpd.models.multiscale_radius_graph import MultiscaleRadiusGraph
from vpd.config import C, M

class GraphBatchNorm1d(torch.nn.Module):
    def __init__(self, in_channels, num_points=None,):
        super(GraphBatchNorm1d, self).__init__()
        self.num_points = num_points
        self.bn = torch.nn.BatchNorm1d(in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, x):
        if self.num_points is None: return self.bn(x)
        _, channel = x.size()
        x = x.view(-1, self.num_points, channel).transpose(2, 1)
        x = self.bn(x)
        return x.transpose(2, 1).reshape(-1, channel).contiguous()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.bn.num_features})'


def MLP(channels, num_points=None, channel_last=True, bias=False):
    if channel_last:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i], bias=bias), GraphBatchNorm1d(channels[i], num_points),
                nn.LeakyReLU(negative_slope=0.2))
            for i in range(1, len(channels))
        ])
    return nn.Sequential(*[
        nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=bias), nn.BatchNorm1d(channels[i]),
            nn.LeakyReLU(negative_slope=0.2))
        for i in range(1, len(channels))
    ])


class DGCN(nn.Module):
    def __init__(self, nf=[], num_nodes=None, num_neighbors=None):
        super(DGCN, self).__init__()

        # Block 0
        self.mlp0a = MLP([nf[0], nf[1]], num_nodes)
        self.mlp0b = MLP([nf[0], nf[1]], num_nodes)

        # Block 1
        self.mlp1a = MLP([nf[1], nf[2]], num_nodes)
        self.mlp1b = MLP([nf[1], nf[2]], num_nodes)

        # Block 2
        self.mlp2a = MLP([nf[2], nf[3]], num_nodes)
        self.mlp2b = MLP([nf[2], nf[3]], num_nodes)

        # Block 3
        self.mlp3a = MLP([nf[3], nf[4]], num_nodes)
        self.mlp3b = MLP([nf[3], nf[4]], num_nodes)
        
        # Block 4
        self.mlp4_cat = MLP([nf[1]+nf[2]+nf[3]+nf[4], 1], num_nodes)
        
        self.bias = nn.Parameter(torch.Tensor(1))
        zeros(self.bias)


        # compute topk neighors for each node
        print('############# MultiscaleRadiusGraph ######################################')
        compute_edge = MultiscaleRadiusGraph(num_nodes, num_neighbors, cache_file=os.path.join('cache/radius_' + C.io.dataset.lower() + '.pt'))
        xyz, edge_index = compute_edge() # edge_index: [2, num_nodes*num_neighbors]
            
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.num_edges = self.edge_index.size(1)
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors

    def forward(self, x):
        # # # reshape CNN feature x from [batch, channel, node] to [batch*node, channel]
        batch, channel, num_nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch * num_nodes, channel)
        # print('input x',  batch, channel, num_nodes, x.size())

        # # # reshape attributes to proper dimensions
        # edge_index: expand from [2, num_edges] to [2, num_edges*batch],  increase the node idx by (number of graphs * number of nodes)
        graph_idx = torch.repeat_interleave(torch.arange(batch, device=self.edge_index.device) * self.num_nodes, self.num_edges).view(1, -1)
        centers, neighbors = self.edge_index.repeat(1, batch) + graph_idx
        # print('centers', centers.shape, centers)
        # print('neighbors', neighbors.shape, neighbors)

        x0 = self.mlp0a(x) + scatter_max(self.mlp0b(x[neighbors] - x[centers]), centers, dim=0)[0]
        x1 = self.mlp1a(x0) + scatter_max(self.mlp1b(x0)[neighbors], centers, dim=0)[0]
        x2 = self.mlp2a(x1) + scatter_max(self.mlp2b(x1)[neighbors], centers, dim=0)[0]
        x3 = self.mlp3a(x2) + scatter_max(self.mlp3b(x2)[neighbors], centers, dim=0)[0]
        x_cat = torch.cat([x0,x1,x2,x3], dim=-1)
        x4 = self.mlp4_cat(x_cat)
        y = x4 + self.bias
        return y
