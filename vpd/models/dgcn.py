import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch_geometric.nn.inits import zeros

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

        # Block cat
        self.mlp_cat = MLP([nf[1]+nf[2]+nf[3]+nf[4], 1], num_nodes)
        self.bias = nn.Parameter(torch.Tensor(1))
        zeros(self.bias)

        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.num_edges = num_nodes*num_neighbors
        print('DGCN num_nodes, num_neighbors, num_edges',  self.num_nodes, self.num_neighbors, self.num_edges)

    def forward(self, x, edge_index):
        # # # reshape CNN feature x from [batch, channel, node] to [batch*node, channel]
        batch, num_nodes, channel = x.shape
        x = x.reshape(batch * num_nodes, channel)

        if batch==1:
            centers, neighbors = edge_index.permute(1, 0, 2).reshape(2, -1)
        else:
            # # # reshape attributes to proper dimensions
            # edge_index: expand from [batch, 2, num_edges] to [2, num_edges*batch],  increase the graph by the number of graphs * number of nodes
            graph_idx = torch.repeat_interleave(torch.arange(batch, device=x.device) * self.num_nodes, self.num_edges).view(1, -1)
            centers, neighbors = edge_index.permute(1,0,2).reshape(2, -1) + graph_idx
        # print('centers', centers.shape, centers)
        # print('neighbors', neighbors.shape, neighbors)

        x0 = self.mlp0a(x) + scatter_max(self.mlp0b(x[neighbors] - x[centers]), centers, dim=0)[0]
        x1 = self.mlp1a(x0) + scatter_max(self.mlp1b(x0)[neighbors], centers, dim=0)[0]
        x2 = self.mlp2a(x1) + scatter_max(self.mlp2b(x1)[neighbors], centers, dim=0)[0]
        x3 = self.mlp3a(x2) + scatter_max(self.mlp3b(x2)[neighbors], centers, dim=0)[0]
        x_cat = torch.cat([x0, x1, x2, x3], dim=-1)
        x_cat = self.mlp_cat(x_cat)
        x_cat = x_cat + self.bias
        return x_cat

