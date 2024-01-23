import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Conv_ReLU_Block(nn.Module):
	def __init__(self):
		super(Conv_ReLU_Block, self).__init__()
		self.conv = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, x):
		return self.relu(self.conv(x))

class BasicBlock(nn.Module):

    def __init__(self, inplanes=32, planes=32):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class SingleGen(nn.Module):
	def __init__(self):
		super(SingleGen, self).__init__()

		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(BasicBlock, 8),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False)
		)
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, x):

		out = self.gen(x)

		return out
		
class LPCNN_INDEP(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()
		self.gt_mean = torch.from_numpy(np.load(gt_mean)[:, np.newaxis, np.newaxis, np.newaxis]).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)[:, np.newaxis, np.newaxis, np.newaxis]).float()

		self.iter_num = 4

		self.alpha = torch.nn.Parameter(torch.ones(1))
		#self.alpha = 1
		'''
		self.gen = nn.Sequential(
				nn.Conv3d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
				nn.ReLU(inplace=True),
				self.make_layer(BasicBlock, 8),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv3d(in_channels=32, out_channels=6, kernel_size=1, stride=1, padding=0, bias=False)
		)
		'''
		self.gen = nn.ModuleList([SingleGen() for i in range(self.iter_num)])

	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, y, dk, mask, ls100):

		batch_size, _, number, x_dim, y_dim, z_dim = y.shape
		_, _, _, w_x_dim, w_y_dim, w_z_dim = dk.shape		
		
		pad_x = w_x_dim - x_dim
		pad_y = w_y_dim - y_dim
		pad_z = w_z_dim - z_dim

		pad_mask = F.pad(mask, (0, pad_z, 0, pad_y, 0, pad_x))


		out = []

		dk = dk.unsqueeze(-1)

		mean = self.gt_mean.to(y.device, dtype=torch.float)
		std = self.gt_std.to(y.device, dtype=torch.float)

		x_est = self.alpha * torch.sum(torch.ifft(dk*torch.rfft(F.pad(y, (0, pad_z, 0, pad_y, 0, pad_x)), 3, normalized=True, onesided=False), 3, normalized=True)[:, :, :, :x_dim, :y_dim, :z_dim, 0], 2)
		
		#den_x_pred = ls100
		#print(self.iter_num)
		#print(self.alpha)

		pn_x_pred = torch.zeros_like(x_est)

		for i in range(self.iter_num):
			
			if i == 0:
				pn_x_pred += x_est

			else:
				
				pn_x_pred = den_x_pred + x_est - self.alpha * torch.sum(torch.ifft(dk*torch.rfft(torch.ifft(torch.sum(dk*torch.rfft(F.pad(den_x_pred, (0, pad_z, 0, pad_y, 0, pad_x)), 3, normalized=True, onesided=False).unsqueeze(2), 1, keepdim=True), 3, normalized=True)[:, :, :, :, :, :, 0] * pad_mask, 3, normalized=True, onesided=False), 3, normalized=True)[:, :, :, :x_dim, :y_dim, :z_dim, 0], 2)
	
			
			x_input = ((pn_x_pred - mean) / std) * mask[:, :, 0, :, :, :]
			x_pred = self.gen[i](x_input)# + x_input
			den_x_pred = ((x_pred * std) + mean) * mask[:, :, 0, :, :, :]
			
			out.append(x_pred)

			#den_x_pred = self.gen[i](pn_x_pred * mask[:, :, 0, :, :, :])
			#den_x_pred = den_x_pred * mask[:, :, 0, :, :, :]

		return x_pred

