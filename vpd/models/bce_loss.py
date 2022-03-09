import os
import numpy as np
import itertools
import torch
import torch.nn as nn

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, label):
        label = label.reshape(-1,1)
        ### separate the pos and neg loss values
        loss = self.loss(output, label)
        loss_pos = loss[label>0.0].sum().float() / label.gt(0.0).sum().float()
        loss_neg = loss[label==0.0].sum().float() / (label.nelement() - label.gt(0.0).sum().float())
        # # # in mutli-gpus case, better to make tensors at least one-dim to stack, otherwise there is a warning
        return loss_pos.unsqueeze(0), loss_neg.unsqueeze(0)

