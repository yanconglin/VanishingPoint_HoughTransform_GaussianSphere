import torch.nn as nn

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, output, label):
        output, label = output.view(-1, 1), label.view(-1, 1)
        loss = self.loss(output, label.view(-1, 1))
        loss_pos = loss[label>0.0].sum().float() / label.gt(0.0).sum().float()
        loss_neg = loss[label==0.0].sum().float() / (label.nelement() - label.gt(0.0).sum().float())
        # print('loss_pos, loss_neg', loss_pos.shape, loss_neg.shape)
        # # # in mutli-gpus case, better to make tensors at least one-dim to stack, otherwise there is a warning
        return loss_pos.unsqueeze(0), loss_neg.unsqueeze(0)
