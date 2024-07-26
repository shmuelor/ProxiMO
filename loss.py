from torch import nn
import torch
import torch.nn.functional as F
from utils import denormalize


class MCLoss(nn.Module):
    def __init__(self, type='l1'):
        """
        Inputs:
            type: 'l1' or 'l2'. default: 'l1'
        """
        super().__init__()
        if type == 'l1':
            self.loss = nn.L1Loss()
        elif type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("type must be 'l1' or 'l2'")

    def forward(self, xhat, A, y, mask):
        batch_size, _, y_dim0, y_dim1, y_dim2 = y.shape
        yhat = torch.zeros_like(y)
        for bn in range(batch_size):
            _, dk_dim0, dk_dim1, dk_dim2 = A[bn].dk.shape
            xhat_pad = F.pad(xhat[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))
            yhat[bn] = A[bn].forward(xhat_pad)[:, :y_dim0, :y_dim1, :y_dim2]
        return self.loss(yhat[mask == 1], y[mask == 1])


class CrossLoss(nn.Module):
    def __init__(self, type='l1'):
        """
        Inputs:
            type: 'l1' or 'l2'. default: 'l1'
        """
        super().__init__()
        if type == 'l1':
            self.loss = nn.L1Loss()
        elif type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("type must be 'l1' or 'l2'")

    def forward(self, xhat_cross, xhat):
        return self.loss(xhat_cross, xhat)
