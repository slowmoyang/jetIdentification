# compatibility with py3
from __future__ import division
from __future__ import print_function
# torch, torchnet and torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


# class WSCLoss(_WeightedLoss):
#     def forward(self, input, target):
#         _assert_no_grad(target)
#         preds_mean = torch.mean(preds[:, 1])
# 	target_wsc_mean = torch.mean(target_wsc)
# 	diff = preds_mean - target_wsc_mean
# 	loss = torch.abs(diff)
#         return loss

class WSCLoss_v1(nn.modules.Module):
    def __init__(self):
        super(WSCLoss_v1, self).__init__()
    def forward(self, output, target):
        output_mean = torch.mean(output[:, 1])
        target_mean = torch.mean(target)
        diff = output_mean - target_mean
        loss = torch.abs(diff)
        return loss


class WSCLoss_v2(nn.modules.Module):
    def __init__(self):
        super(WSCLoss_v1, self).__init__()
    def forward(self, output, target):
        diff = output[:, 1] - target
        l1_distance = torch.abs(diff)
        # sum or mean?
        loss = torch.sum(l1_distance)
        return loss


def calc_wsc_loss(preds, target_wsc):
    preds_mean = torch.mean(preds[:, 1])
    target_wsc_mean = torch.mean(target_wsc)
    diff = preds_mean - target_wsc_mean
    loss = torch.abs(diff)
    return loss

