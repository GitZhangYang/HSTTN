
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FilterMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(FilterMSELoss, self).__init__()

    def forward(self, pred, gold, raw, col_names):
        # Remove bad input
        cond1 = raw[:, :, :, col_names["Patv"]] < 0

        cond2 = raw[:, :, :, col_names["Pab1"]] > 89
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab2"]] > 89)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Pab3"]] > 89)

        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Wdir"]] < -180)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Wdir"]] > 180)
        cond2 = torch.logical_or(cond2,
                                  raw[:, :, :, col_names["Ndir"]] < -720)
        cond2 = torch.logical_or(cond2, raw[:, :, :, col_names["Ndir"]] > 720)
        cond2 = torch.logical_or(cond2, cond1)

        cond3 = raw[:, :, :, col_names["Patv"]] == 0
        cond3 = torch.logical_and(cond3,
                                   raw[:, :, :, col_names["Wspd"]] > 2.5)
        cond3 = torch.logical_or(cond3, cond2)

        cond = torch.logical_not(cond3)
        cond = cond.float()

        return torch.mean(F.mse_loss(pred, gold, reduction='none') * cond)


class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()

    def forward(self, pred, gold, raw=None, col_names=None):
        return F.mse_loss(pred, gold)

class RMSE(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, pred: torch.Tensor, target: torch.Tensor,raw,col_names):
        return torch.sqrt(
            torch.mean(torch.square(
                (pred - target).abs()))
        )
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target,raw=None,col_names=None):
        return torch.mean((pred - target).abs())

class MAPE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight=None):
        ape = (pred - target).abs() / (target.abs() + self.eps)
        if weight is None:
            return torch.mean(ape)
        else:
            return (ape * weight / (weight.sum())).sum()


class HuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, gold, raw, col_names):
        loss = F.smooth_l1_loss(pred, gold, reduction='mean', delta=self.delta)
        return loss


class SmoothMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(SmoothMSELoss, self).__init__()
        self.smooth_win = kwargs["smooth_win"]

    def forward(self, pred, gold, raw, col_names):
        gold = F.avg_pool1d(
            gold, self.smooth_win, stride=1, padding="SAME", exclusive=False)
        loss = F.mse_loss(pred, gold)
        return loss


def masked_mae_loss(y_pred, y_true,mean,std):
    mask = (y_true != 0).float()
    #print("mask shape:{}, mask mean:{}".format(mask.shape, mask.mean()))
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true,mean,std):
    mean = np.expand_dims(mean[:, :, -1], 0)
    std = np.expand_dims(std[:, :, -1], 0)
    invalid = torch.Tensor(((-1) - mean)/std).to(y_true.device)
    #print("invalid shape:{}".format(invalid.shape))
    mask = (y_true != invalid).float()
    #print("mask shape:{}, mask mean:{}".format(mask.shape, mask.mean()))
    #mask = mask.float
    mask /= mask.mean()
    loss = F.mse_loss(y_pred,y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
