from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

from proximal_sti.lib.model.lpcnn.unet3d.models import ResidualUNet3D
from proximal_sti.lib.model.lpcnn.lpcnn_resunet import pad_for_unet, unpad_for_unet

class LPCNN(nn.Module):
    def __init__(self, gt_mean=0., gt_std=1., unet=True, use_pinv=False, norm=False):
        super().__init__()
        self.register_buffer('gt_mean', torch.tensor(gt_mean).float())
        self.register_buffer('gt_std', torch.tensor(gt_std).float())

        self.iter_num = 3

        self.alpha = torch.nn.Parameter(torch.ones(1) * 4)

        self.unet = unet
        if self.unet:
            self.gen = ResidualUNet3D(in_channels=1, out_channels=1)
        else:
            self.gen = nn.Sequential(
            		nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            		nn.ReLU(inplace=True),
            		self.make_layer(wBasicBlock, 8),
            		nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            		nn.ReLU(inplace=True),
            		nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            		nn.ReLU(inplace=True),
            		nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
            )

        self.use_pinv = use_pinv
        self.norm = norm
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, y, A, mask):
        """
        y: batch, nori, w, h, d
        A: list of class QsmOperator
            A[i].dk: nori, w, h, d. torch.Tensor
            A[i].mask: w, h, d. torch.Tensor
        """
        mean = self.gt_mean.to(y.device, dtype=torch.float)
        std = self.gt_std.to(y.device, dtype=torch.float)

        batch_size, _, y_dim0, y_dim1, y_dim2 = y.shape

        # TODO - adapt to multiple orientations 
        mask = mask.unsqueeze(1)
        den_x_pred = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
        x_est = torch.zeros_like(den_x_pred, dtype=y.dtype, device=y.device)
        for bn in range(batch_size):
            
            _, dk_dim0, dk_dim1, dk_dim2 = A[bn].dk.shape

            yy_pad = F.pad(y[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))

            # print(torch.linalg.norm(A[bn].adjoint(yy_pad)[:y_dim0, :y_dim1, :y_dim2]).item())
            # print(torch.linalg.norm(_ifft(A[bn].dk.unsqueeze(-1) * _rfft(F.pad(y[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))))[:, :y_dim0, :y_dim1, :y_dim2, 0]).item())
            
            x_est[bn, 0] = self.alpha * A[bn].adjoint(yy_pad)[:y_dim0, :y_dim1, :y_dim2]
            # x_est[bn] = self.alpha * _ifft(A[bn].dk.unsqueeze(-1) * _rfft(F.pad(y[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))))[:, :y_dim0, :y_dim1, :y_dim2, 0]

        for i in range(self.iter_num):

            if i == 0:
                if not self.use_pinv:
                    den_x_pred[bn, 0] = torch.zeros_like(x_est[bn, 0]) 
                else:
                    den_x_pred[bn, 0] = A[bn].pinv(yy_pad, epsilon=1e-2)[:y_dim0, :y_dim1, :y_dim2]
                
                den_x_pred += x_est # TODO - this is only when nori is 1, otherwise we need to sum x_est over all the orientations and add to den_x_pred
            
            else:
                 
                for bn in range(batch_size):
                    _, dk_dim0, dk_dim1, dk_dim2 = A[bn].dk.shape
                    
                    den_x_pred_pad = F.pad(den_x_pred, (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))

                    # print(torch.linalg.norm(A[bn].gram(den_x_pred_pad[bn, 0])[:y_dim0, :y_dim1, :y_dim2]).item())
                    # print(torch.linalg.norm(_ifft(A[bn].dk.unsqueeze(-1) * A[bn].dk.unsqueeze(-1) * _rfft(F.pad(den_x_pred[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))))[:, :y_dim0, :y_dim1, :y_dim2, 0]).item())

                    den_x_pred[bn, 0] = den_x_pred[bn, 0] + x_est[bn, 0] - self.alpha * A[bn].gram(den_x_pred_pad[bn, 0])[:y_dim0, :y_dim1, :y_dim2]

                    # den_x_pred[bn, :, :, :, :] = den_x_pred[bn, :, :, :, :] + x_est[bn, 0] - self.alpha * _ifft(A[bn].dk.unsqueeze(-1) * A[bn].dk.unsqueeze(-1) * _rfft(F.pad(den_x_pred[bn], (0, dk_dim2-y_dim2, 0, dk_dim1-y_dim1, 0, dk_dim0-y_dim0))))[:, :y_dim0, :y_dim1, :y_dim2, 0]

            # normalize
            if self.norm:
                 den_x_pred = (den_x_pred - mean) / std
            
            # call the network
            x_input = den_x_pred * mask
            if self.unet:
                x_input_padded = pad_for_unet(x_input, len(self.gen.encoders))
            else:
                x_input_padded = x_input
            
            x_pred = self.gen(x_input_padded)
            # x_pred = x_input_padded
            
            if self.unet:
                x_pred = unpad_for_unet(x_pred, x_input, len(self.gen.encoders)) 
            
            # denormalize
            if self.norm:
                 den_x_pred = (x_pred * std) + mean
            den_x_pred = den_x_pred * mask

        return x_pred


class wBasicBlock(nn.Module):

	def __init__(self, inplanes=32, planes=32, dropout_rate=0.5):
		super(wBasicBlock, self).__init__()
		self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
		self.bn1 = nn.BatchNorm3d(planes)
		self.relu = nn.ReLU(inplace=True)
	
		self.dropout = nn.Dropout3d(p=dropout_rate)
	
		self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
		self.bn2 = nn.BatchNorm3d(planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.dropout(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += residual
		out = self.relu(out)

		return out

def _rfft(x):
    try:
        # torch==1.2.0
        x = torch.rfft(x, 3, normalized=True, onesided=False)
    except:
        # torch==1.10.0
        x = torch.fft.fftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x

def _ifft(x):
    try:
        # torch==1.2.0
        x = torch.ifft(x, 3, normalized=True)
    except:
        # torch==1.10.0
        x = torch.view_as_complex(x)
        x = torch.fft.ifftn(x, dim=(-3, -2, -1), norm='ortho')
        x = torch.stack((x.real, x.imag), dim=-1)
    return x