import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
import numpy as np

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

class VDSRR(nn.Module):
	def __init__(self, gt_mean, gt_std):
		super().__init__()

		self.gt_mean = torch.from_numpy(np.load(gt_mean)[:, np.newaxis, np.newaxis, np.newaxis]).float()
		self.gt_std = torch.from_numpy(np.load(gt_std)[:, np.newaxis, np.newaxis, np.newaxis]).float()

		self.alpha = torch.nn.Parameter(torch.ones(1)*6)

		self.residual_layer = self.make_layer(BasicBlock, 8)
		self.input = nn.Conv3d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv3d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		'''
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))
		'''		
	def make_layer(self, block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, y, dk, mask, ls100):		 
		
		batch_size, _, x_dim, y_dim, z_dim, number = y.shape
		
		dk_batch = []
		dim1_batch = []
		dim2_batch = []
		dim3_batch = []
			
		x_est = torch.zeros_like(y[:, :, :, :, :, 0]).expand(batch_size, 6, -1, -1, -1).clone()
		
		for b_n in range(batch_size):
			dk_batch.append([])
			dim1_batch.append([])
			dim2_batch.append([])
			dim3_batch.append([])

			dk_list = dk[b_n].split(' ')[:-1]

			for num in range(number):
				dk_batch[-1].append(torch.from_numpy(np.moveaxis(np.load(dk_list[num])[:, :, :, :, np.newaxis], 3, 0)).to(y.device, dtype=torch.float))

				dim1_batch[-1].append(dk_batch[-1][-1].shape[1])
				dim2_batch[-1].append(dk_batch[-1][-1].shape[2])
				dim3_batch[-1].append(dk_batch[-1][-1].shape[3])			

				x_est[b_n, :, :, :, :] += (self.alpha/number) * torch.ifft(dk_batch[-1][num] * torch.rfft(F.pad(y[b_n, :, :, :, :, num], (0, dim3_batch[-1][num]-z_dim, 0, dim2_batch[-1][num]-y_dim, 0, dim1_batch[-1][num]-x_dim)), 3, normalized=True, onesided=False), 3, normalized=True)[:, :x_dim, :y_dim, :z_dim, 0]
		
		norm_x_est = ((x_est - self.gt_mean.to(y.device, dtype=torch.float)) / self.gt_std.to(y.device, dtype=torch.float)) * mask[:, :, :, :, :, 0]
		
		residual = ls100
		#residual = x_est

		out = self.relu(self.input(ls100))
		#out = self.relu(self.input(x_est))
		out = self.residual_layer(out)
		out = self.output(out)

		out = torch.add(out, residual)
		
		return out
